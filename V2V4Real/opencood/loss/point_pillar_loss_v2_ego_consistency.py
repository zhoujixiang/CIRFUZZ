import torch
import torch.nn.functional as F

from opencood.loss.point_pillar_loss_v2 import (
    PointPillarLossV2,
    add_sin_difference,
    get_direction_target,
    sigmoid_focal_loss,
    softmax_cross_entropy_with_logits,
    weighted_smooth_l1_loss,
)


class PointPillarLossV2EgoConsistency(PointPillarLossV2):
    def __init__(self, args):
        super(PointPillarLossV2EgoConsistency, self).__init__(args)

        self.ego_aux_weight = args.get('ego_aux_weight', 0.5)
        self.preserve_cls_weight = args.get('preserve_cls_weight', 0.2)
        self.preserve_reg_weight = args.get('preserve_reg_weight', 0.2)
        self.preserve_neg_weight = args.get('preserve_neg_weight', 0.1)
        self.gate_weight = args.get('gate_weight', 0.01)

        self.ego_positive_thresh = args.get('ego_positive_thresh', 0.35)
        self.ego_negative_thresh = args.get('ego_negative_thresh', 0.10)
        self.cls_margin = args.get('cls_margin', 0.02)
        self.reg_margin = args.get('reg_margin', 0.0)
        self.neg_margin = args.get('neg_margin', 0.02)

    @staticmethod
    def _masked_mean(values, mask):
        if mask.any():
            return values[mask].mean()
        return values.new_zeros(())

    def _compute_detection_loss(self, output_dict, target_dict):
        psm = output_dict['psm'] if 'psm' in output_dict else \
            output_dict['cls_preds']
        rm = output_dict['rm'] if 'rm' in output_dict else \
            output_dict['reg_preds']
        dir_preds = output_dict.get('dir_preds', output_dict.get('dm'))

        batch_size = target_dict['pos_equal_one'].shape[0]
        cls_labels = target_dict['pos_equal_one'].view(batch_size, -1, 1)
        positives = cls_labels > 0
        negatives = target_dict['neg_equal_one'].view(batch_size, -1, 1) > 0
        pos_normalizer = positives.sum(1, keepdim=True).float()

        cls_weights = positives * self.pos_cls_weight + negatives * 1.0
        cls_weights = cls_weights.float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_preds = psm.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
        cls_targets = cls_labels.squeeze(dim=-1)
        one_hot_targets = torch.zeros(*list(cls_targets.shape), 2,
                                      dtype=cls_preds.dtype,
                                      device=cls_preds.device)
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = sigmoid_focal_loss(cls_preds,
                                          one_hot_targets,
                                          weights=cls_weights,
                                          **self.cls)
        cls_loss = cls_loss_src.sum() * self.cls['weight'] / batch_size

        reg_weights = positives.float() / torch.clamp(pos_normalizer, min=1.0)
        reg_preds = rm.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
        reg_targets = target_dict['targets'].view(batch_size, -1, 7)
        reg_preds, reg_targets = add_sin_difference(reg_preds, reg_targets)
        reg_loss = weighted_smooth_l1_loss(reg_preds,
                                           reg_targets,
                                           weights=reg_weights,
                                           sigma=self.reg['sigma'])
        reg_loss = reg_loss.sum() * self.reg['weight'] / batch_size

        dir_loss = cls_loss.new_zeros(())
        if self.dir and dir_preds is not None:
            dir_targets = get_direction_target(target_dict['targets'].view(
                batch_size, -1, 7), self.dir)
            dir_logits = dir_preds.permute(0, 2, 3, 1).contiguous().view(
                batch_size, -1, self.dir['args']['num_bins'])
            dir_loss_src = softmax_cross_entropy_with_logits(
                dir_logits.contiguous().view(-1, self.dir['args']['num_bins']),
                dir_targets.contiguous().view(-1, self.dir['args']['num_bins']))
            dir_loss = dir_loss_src.flatten() * reg_weights.flatten()
            dir_loss = dir_loss.sum() * self.dir['weight'] / batch_size

        total_loss = cls_loss + reg_loss + dir_loss
        return total_loss, reg_loss, cls_loss, dir_loss

    def _per_anchor_reg_error(self, rm, targets):
        batch_size = rm.shape[0]
        rm = rm.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
        targets = targets.view(batch_size, -1, 7)
        box_preds_sin, reg_targets_sin = add_sin_difference(rm, targets)
        return weighted_smooth_l1_loss(box_preds_sin,
                                       reg_targets_sin,
                                       weights=None,
                                       sigma=self.reg['sigma']).sum(dim=-1)

    def forward(self, output_dict, target_dict):
        fused_total, fused_reg_loss, fused_conf_loss, fused_dir_loss = \
            self._compute_detection_loss(output_dict, target_dict)

        ego_total = fused_total.new_zeros(())
        preserve_cls_loss = fused_total.new_zeros(())
        preserve_reg_loss = fused_total.new_zeros(())
        preserve_neg_loss = fused_total.new_zeros(())
        gate_loss = fused_total.new_zeros(())

        if 'ego_psm' in output_dict and 'ego_rm' in output_dict:
            ego_total, _, _, _ = self._compute_detection_loss(
                {'psm': output_dict['ego_psm'], 'rm': output_dict['ego_rm']},
                target_dict)

            batch_size = output_dict['psm'].shape[0]
            positives = target_dict['pos_equal_one'].view(batch_size, -1) > 0
            negatives = target_dict['neg_equal_one'].view(batch_size, -1) > 0

            ego_prob = torch.sigmoid(
                output_dict['ego_psm'].permute(0, 2, 3, 1).contiguous()
            ).view(batch_size, -1)
            fused_prob = torch.sigmoid(
                output_dict['psm'].permute(0, 2, 3, 1).contiguous()
            ).view(batch_size, -1)

            protected_pos = positives & \
                (ego_prob.detach() > self.ego_positive_thresh)
            protected_neg = negatives & \
                (ego_prob.detach() < self.ego_negative_thresh)

            cls_drop = F.relu(ego_prob.detach() - fused_prob - self.cls_margin)
            preserve_cls_loss = self._masked_mean(cls_drop, protected_pos)

            targets = target_dict['targets']
            fused_reg_error = self._per_anchor_reg_error(output_dict['rm'],
                                                         targets)
            ego_reg_error = self._per_anchor_reg_error(output_dict['ego_rm'],
                                                       targets).detach()
            reg_worse = F.relu(fused_reg_error - ego_reg_error -
                               self.reg_margin)
            preserve_reg_loss = self._masked_mean(reg_worse, protected_pos)

            neg_rise = F.relu(fused_prob - ego_prob.detach() - self.neg_margin)
            preserve_neg_loss = self._masked_mean(neg_rise, protected_neg)

        if 'fusion_gate' in output_dict:
            gate_loss = output_dict['fusion_gate'].mean()

        total_loss = fused_total + \
            self.ego_aux_weight * ego_total + \
            self.preserve_cls_weight * preserve_cls_loss + \
            self.preserve_reg_weight * preserve_reg_loss + \
            self.preserve_neg_weight * preserve_neg_loss + \
            self.gate_weight * gate_loss

        self.loss_dict.update({
            'total_loss': total_loss.item(),
            'reg_loss': fused_reg_loss.item(),
            'conf_loss': fused_conf_loss.item(),
            'dir_loss': fused_dir_loss.item(),
            'ego_aux_loss': ego_total.item(),
            'preserve_cls_loss': preserve_cls_loss.item(),
            'preserve_reg_loss': preserve_reg_loss.item(),
            'preserve_neg_loss': preserve_neg_loss.item(),
            'gate_loss': gate_loss.item()
        })

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']
        dir_loss = self.loss_dict['dir_loss']
        ego_aux_loss = self.loss_dict['ego_aux_loss']
        preserve_cls_loss = self.loss_dict['preserve_cls_loss']
        preserve_reg_loss = self.loss_dict['preserve_reg_loss']
        preserve_neg_loss = self.loss_dict['preserve_neg_loss']
        gate_loss = self.loss_dict['gate_loss']

        message = (
            "[epoch %d][%d/%d] || Loss: %.4f || Conf: %.4f || Loc: %.4f"
            " || Dir: %.4f || EgoAux: %.4f || KeepCls: %.4f"
            " || KeepReg: %.4f || KeepNeg: %.4f || Gate: %.4f"
        ) % (epoch, batch_id + 1, batch_len,
             total_loss, conf_loss, reg_loss, dir_loss,
             ego_aux_loss, preserve_cls_loss, preserve_reg_loss,
             preserve_neg_loss, gate_loss)

        if pbar is None:
            print(message)
        else:
            pbar.set_description(message)

        step = epoch * batch_len + batch_id
        writer.add_scalar('Regression_loss', reg_loss, step)
        writer.add_scalar('Confidence_loss', conf_loss, step)
        writer.add_scalar('Dir_loss', dir_loss, step)
        writer.add_scalar('Ego_aux_loss', ego_aux_loss, step)
        writer.add_scalar('Preserve_cls_loss', preserve_cls_loss, step)
        writer.add_scalar('Preserve_reg_loss', preserve_reg_loss, step)
        writer.add_scalar('Preserve_neg_loss', preserve_neg_loss, step)
        writer.add_scalar('Gate_loss', gate_loss, step)
