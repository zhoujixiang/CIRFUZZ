
import numpy as np
import torch
import torch.nn as nn
from opencood.utils.common_utils_ext import limit_period


class PointPillarLossV2(nn.Module):
    def __init__(self, args):
        super(PointPillarLossV2, self).__init__()
        self.pos_cls_weight = args['pos_cls_weight']
        self.cls = args['cls']
        self.reg = args['reg']

        if 'dir' in args:
            self.dir = args['dir']
        else:
            self.dir = None

        if 'iou' in args:
            raise NotImplementedError("IoU loss not supported in this version")
        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix=""):
        batch_size = (
            target_dict['pos_equal_one'].shape[0]
            if 'record_len' not in output_dict
            else int(output_dict['record_len'].sum())
        )

        cls_labels = target_dict['pos_equal_one'].view(batch_size, -1, 1)
        positives = cls_labels > 0
        negatives = target_dict['neg_equal_one'].view(batch_size, -1, 1) > 0
        pos_normalizer = positives.sum(1, keepdim=True).float()

        for key_src, key_dst in [('psm', 'cls_preds'), ('rm', 'reg_preds'),
                                  ('dm', 'dir_preds')]:
            if f'{key_src}{suffix}' in output_dict:
                output_dict[f'{key_dst}{suffix}'] = output_dict[f'{key_src}{suffix}']

        total_loss = 0

        cls_preds = (output_dict[f'cls_preds{suffix}']
                     .permute(0, 2, 3, 1).contiguous()
                     .view(batch_size, -1, 1))
        cls_weights = positives * self.pos_cls_weight + negatives * 1.0
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss = sigmoid_focal_loss(cls_preds, cls_labels,
                                      weights=cls_weights, **self.cls)
        cls_loss = cls_loss.sum() * self.cls['weight'] / batch_size

        reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
        reg_preds = (output_dict[f'reg_preds{suffix}']
                     .permute(0, 2, 3, 1).contiguous()
                     .view(batch_size, -1, 7))
        reg_targets = target_dict['targets'].view(batch_size, -1, 7)
        reg_preds, reg_targets = add_sin_difference(reg_preds, reg_targets)
        reg_loss = weighted_smooth_l1_loss(reg_preds, reg_targets,
                                           weights=reg_weights,
                                           sigma=self.reg['sigma'])
        reg_loss = reg_loss.sum() * self.reg['weight'] / batch_size

        if self.dir:
            dir_targets = get_direction_target(
                target_dict['targets'].view(batch_size, -1, 7),
                self.dir)
            dir_logits = (output_dict[f'dir_preds{suffix}']
                          .permute(0, 2, 3, 1).contiguous()
                          .view(batch_size, -1, 2))
            dir_loss = softmax_cross_entropy_with_logits(
                dir_logits.contiguous().view(-1, 2),
                dir_targets.contiguous().view(-1, 2))
            dir_loss = dir_loss.flatten() * reg_weights.flatten()
            dir_loss = dir_loss.sum() * self.dir['weight'] / batch_size
            total_loss += dir_loss
            self.loss_dict.update({'dir_loss': dir_loss.item()})

        total_loss += reg_loss + cls_loss

        self.loss_dict.update({
            'total_loss': total_loss.item(),
            'reg_loss': reg_loss.item(),
            'cls_loss': cls_loss.item()
        })

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix="", pbar=None):
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)

        message = (f"[epoch {epoch}][{batch_id + 1}/{batch_len}]{suffix} "
                   f"|| Loss: {total_loss:.4f} || Conf Loss: {cls_loss:.4f} "
                   f"|| Loc Loss: {reg_loss:.4f} || Dir Loss: {dir_loss:.4f}")
        if pbar is None:
            print(message)
        else:
            pbar.set_description(message)

        if writer is not None:
            writer.add_scalar(f'Regression_loss{suffix}', reg_loss,
                              epoch * batch_len + batch_id)
            writer.add_scalar(f'Confidence_loss{suffix}', cls_loss,
                              epoch * batch_len + batch_id)
            writer.add_scalar(f'Dir_loss{suffix}', dir_loss,
                              epoch * batch_len + batch_id)


def add_sin_difference(boxes1, boxes2, dim=6):
    rad_pred = torch.sin(boxes1[..., dim:dim + 1]) * \
               torch.cos(boxes2[..., dim:dim + 1])
    rad_tg = torch.cos(boxes1[..., dim:dim + 1]) * \
             torch.sin(boxes2[..., dim:dim + 1])
    boxes1 = torch.cat([boxes1[..., :dim], rad_pred, boxes1[..., dim + 1:]], dim=-1)
    boxes2 = torch.cat([boxes2[..., :dim], rad_tg, boxes2[..., dim + 1:]], dim=-1)
    return boxes1, boxes2


def get_direction_target(reg_targets, dir_cfg):
    num_bins = dir_cfg['args']['num_bins']
    dir_offset = dir_cfg['args']['dir_offset']
    anchor_yaw = np.deg2rad(np.array(dir_cfg['args']['anchor_yaw']))
    anchor_yaw_map = torch.from_numpy(anchor_yaw).view(1, -1, 1).to(reg_targets.device)

    HWA = reg_targets.shape[1]
    anchor_map = anchor_yaw_map.repeat(1, HWA // anchor_yaw_map.shape[1], 1)
    rot_gt = reg_targets[..., -1] + anchor_map[..., -1]
    offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    dir_cls_targets = one_hot(dir_cls_targets, num_bins)
    return dir_cls_targets


def one_hot(tensor, num_bins, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(
        *list(tensor.shape), num_bins, dtype=dtype, device=tensor.device)
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot


def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss


def weighted_smooth_l1_loss(preds, targets, sigma=3.0, weights=None):
    diff = preds - targets
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)
    loss = (abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) +
            (abs_diff - 0.5 / (sigma ** 2)) * (1.0 - abs_diff_lt_1))
    if weights is not None:
        loss *= weights
    return loss


def sigmoid_focal_loss(preds, targets, weights=None, **kwargs):
    gamma = kwargs['gamma']
    alpha = kwargs['alpha']

    per_entry_cross_ent = (torch.clamp(preds, min=0) -
                           preds * targets.type_as(preds) +
                           torch.log1p(torch.exp(-torch.abs(preds))))
    prediction_probabilities = torch.sigmoid(preds)
    p_t = (targets * prediction_probabilities +
           (1 - targets) * (1 - prediction_probabilities))
    modulating_factor = torch.pow(1.0 - p_t, gamma)
    alpha_weight_factor = targets * alpha + (1 - targets) * (1 - alpha)

    loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
    if weights is not None:
        loss *= weights
    return loss
