import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.loss.point_pillar_loss import PointPillarLoss


class PointPillarEgoConsistencyLoss(PointPillarLoss):

    def __init__(self, args):
        super(PointPillarEgoConsistencyLoss, self).__init__(args)

        self.ego_aux_weight = args.get('ego_aux_weight', 0.5)
        self.preserve_cls_weight = args.get('preserve_cls_weight', 0.2)
        self.preserve_reg_weight = args.get('preserve_reg_weight', 0.2)
        self.preserve_neg_weight = args.get('preserve_neg_weight', 0.1)
        self.gate_weight = args.get('gate_weight', 0.0)

        self.ego_positive_thresh = args.get('ego_positive_thresh', 0.35)
        self.ego_negative_thresh = args.get('ego_negative_thresh', 0.10)
        self.cls_margin = args.get('cls_margin', 0.02)
        self.reg_margin = args.get('reg_margin', 0.0)
        self.neg_margin = args.get('neg_margin', 0.02)

    def _compute_detection_loss(self, psm, rm, target_dict):
        targets = target_dict['targets']
        cls_preds = psm.permute(0, 2, 3, 1).contiguous()

        box_cls_labels = target_dict['pos_equal_one']
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()
        box_cls_labels_neg = target_dict['neg_equal_one']
        box_cls_labels_neg = box_cls_labels_neg.view(psm.shape[0],
                                                     -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels_neg > 0

        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_targets = box_cls_labels.unsqueeze(dim=-1).squeeze(dim=-1)
        one_hot_targets = torch.zeros(*list(cls_targets.shape), 2,
                                      dtype=cls_preds.dtype,
                                      device=cls_preds.device)
        one_hot_targets.scatter_(-1,
                                 cls_targets.unsqueeze(dim=-1).long(),
                                 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = self.cls_loss_func(cls_preds,
                                          one_hot_targets,
                                          weights=cls_weights)
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        rm = rm.permute(0, 2, 3, 1).contiguous().view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin,
                                          reg_targets_sin,
                                          weights=reg_weights)
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss = reg_loss * self.reg_coe

        total_loss = reg_loss + conf_loss
        return total_loss, reg_loss, conf_loss

    def _per_anchor_reg_error(self, rm, targets):
        rm = rm.permute(0, 2, 3, 1).contiguous().view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)
        return self.reg_loss_func(box_preds_sin,
                                  reg_targets_sin,
                                  weights=None).sum(dim=-1)

    @staticmethod
    def _masked_mean(values, mask):
        if mask.any():
            return values[mask].mean()
        return values.new_zeros(())

    def forward(self, output_dict, target_dict):
        fused_total, fused_reg_loss, fused_conf_loss = \
            self._compute_detection_loss(output_dict['psm'],
                                         output_dict['rm'],
                                         target_dict)

        ego_total = fused_total.new_zeros(())
        preserve_cls_loss = fused_total.new_zeros(())
        preserve_reg_loss = fused_total.new_zeros(())
        preserve_neg_loss = fused_total.new_zeros(())
        gate_loss = fused_total.new_zeros(())
        protected_pos_ratio = fused_total.new_zeros(())
        protected_neg_ratio = fused_total.new_zeros(())

        if 'ego_psm' in output_dict and 'ego_rm' in output_dict:
            ego_total, _, _ = self._compute_detection_loss(output_dict['ego_psm'],
                                                           output_dict['ego_rm'],
                                                           target_dict)

            targets = target_dict['targets']
            positives = target_dict['pos_equal_one'].view(output_dict['psm'].shape[0],
                                                          -1) > 0
            negatives = target_dict['neg_equal_one'].view(output_dict['psm'].shape[0],
                                                          -1) > 0

            ego_prob = torch.sigmoid(
                output_dict['ego_psm'].permute(0, 2, 3, 1).contiguous()
            ).view(output_dict['psm'].shape[0], -1)
            fused_prob = torch.sigmoid(
                output_dict['psm'].permute(0, 2, 3, 1).contiguous()
            ).view(output_dict['psm'].shape[0], -1)

            protected_pos = positives & \
                (ego_prob.detach() > self.ego_positive_thresh)
            protected_neg = negatives & \
                (ego_prob.detach() < self.ego_negative_thresh)

            protected_pos_ratio = protected_pos.float().mean()
            protected_neg_ratio = protected_neg.float().mean()

            cls_drop = F.relu(ego_prob.detach() - fused_prob - self.cls_margin)
            preserve_cls_loss = self._masked_mean(cls_drop, protected_pos)

            fused_reg_error = self._per_anchor_reg_error(output_dict['rm'],
                                                         targets)
            ego_reg_error = self._per_anchor_reg_error(output_dict['ego_rm'],
                                                       targets).detach()
            reg_worse = F.relu(fused_reg_error - ego_reg_error -
                               self.reg_margin)
            preserve_reg_loss = self._masked_mean(reg_worse, protected_pos)

            neg_rise = F.relu(fused_prob - ego_prob.detach() - self.neg_margin)
            preserve_neg_loss = self._masked_mean(neg_rise, protected_neg)

        if 'fusion_gate' in output_dict and self.gate_weight > 0:
            gate_loss = output_dict['fusion_gate'].mean()

        total_loss = fused_total + \
            self.ego_aux_weight * ego_total + \
            self.preserve_cls_weight * preserve_cls_loss + \
            self.preserve_reg_weight * preserve_reg_loss + \
            self.preserve_neg_weight * preserve_neg_loss + \
            self.gate_weight * gate_loss

        self.loss_dict.update({
            'total_loss': total_loss,
            'reg_loss': fused_reg_loss,
            'conf_loss': fused_conf_loss,
            'ego_aux_loss': ego_total,
            'preserve_cls_loss': preserve_cls_loss,
            'preserve_reg_loss': preserve_reg_loss,
            'preserve_neg_loss': preserve_neg_loss,
            'gate_loss': gate_loss,
            'protected_pos_ratio': protected_pos_ratio,
            'protected_neg_ratio': protected_neg_ratio
        })

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']
        ego_aux_loss = self.loss_dict['ego_aux_loss']
        preserve_cls_loss = self.loss_dict['preserve_cls_loss']
        preserve_reg_loss = self.loss_dict['preserve_reg_loss']
        preserve_neg_loss = self.loss_dict['preserve_neg_loss']
        gate_loss = self.loss_dict['gate_loss']
        protected_pos_ratio = self.loss_dict['protected_pos_ratio']
        protected_neg_ratio = self.loss_dict['protected_neg_ratio']

        message = (
            "[epoch %d][%d/%d], || Loss: %.4f || Conf: %.4f || Loc: %.4f"
            " || EgoAux: %.4f || KeepCls: %.4f || KeepReg: %.4f"
            " || KeepNeg: %.4f || Gate: %.4f"
        ) % (epoch, batch_id + 1, batch_len,
             total_loss.item(), conf_loss.item(), reg_loss.item(),
             ego_aux_loss.item(), preserve_cls_loss.item(),
             preserve_reg_loss.item(), preserve_neg_loss.item(),
             gate_loss.item())

        if pbar is None:
            print(message)
        else:
            pbar.set_description(message)

        step = epoch * batch_len + batch_id
        writer.add_scalar('Regression_loss', reg_loss.item(), step)
        writer.add_scalar('Confidence_loss', conf_loss.item(), step)
        writer.add_scalar('Ego_aux_loss', ego_aux_loss.item(), step)
        writer.add_scalar('Preserve_cls_loss', preserve_cls_loss.item(), step)
        writer.add_scalar('Preserve_reg_loss', preserve_reg_loss.item(), step)
        writer.add_scalar('Preserve_neg_loss', preserve_neg_loss.item(), step)
        writer.add_scalar('Gate_loss', gate_loss.item(), step)
        writer.add_scalar('Protected_pos_ratio',
                          protected_pos_ratio.item(),
                          step)
        writer.add_scalar('Protected_neg_ratio',
                          protected_neg_ratio.item(),
                          step)
