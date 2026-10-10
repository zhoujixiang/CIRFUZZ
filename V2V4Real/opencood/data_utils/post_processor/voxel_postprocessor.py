import math
import sys

import numpy as np
import torch
from torch.nn.functional import sigmoid
import torch.nn.functional as F

from opencood.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor
from opencood.utils import box_utils, common_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils


class VoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train):
        super(VoxelPostprocessor, self).__init__(anchor_params, train)
        self.anchor_num = self.params['anchor_args']['num']
        self.safe_log_target = self.params.get('safe_log_target', False)
        self.safe_log_eps = self.params.get('safe_log_eps', 1e-6)
        
    def generate_anchor_box(self):
        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']

        l = self.params['anchor_args']['l']
        w = self.params['anchor_args']['w']
        h = self.params['anchor_args']['h']
        r = self.params['anchor_args']['r']

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]

        vh = self.params['anchor_args']['vh']
        vw = self.params['anchor_args']['vw']

        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]

        if 'feature_stride' in self.params['anchor_args']:
            feature_stride = self.params['anchor_args']['feature_stride']
        else:
            feature_stride = 2

        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)

        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num)
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0

        w = np.ones_like(cx) * w
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h

        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]

        if self.params['order'] == 'hwl':
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1)
        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')

        return anchors

    def generate_label(self, **kwargs):
        assert self.params['order'] == 'hwl', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        gt_box_center = kwargs['gt_box_center']
        anchors = kwargs['anchors']
        masks = kwargs['mask']

        feature_map_shape = anchors.shape[:2]

        anchors = anchors.reshape(-1, 7)
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        targets = np.zeros((*feature_map_shape, self.anchor_num * 7))

        gt_box_center_valid = gt_box_center[masks == 1]
        gt_box_corner_valid = \
            box_utils.boxes_to_corners_3d(gt_box_center_valid,
                                          self.params['order'])
        anchors_corner = \
            box_utils.boxes_to_corners_3d(anchors,
                                          order=self.params['order'])
        anchors_standup_2d = \
            box_utils.corner2d_to_standup_box(anchors_corner)
        gt_standup_2d = \
            box_utils.corner2d_to_standup_box(gt_box_corner_valid)

        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        id_highest = np.argmax(iou.T, axis=1)
        id_highest_gt = np.arange(iou.T.shape[0])
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        id_pos, id_pos_gt = \
            np.where(iou >
                     self.params['target_args']['pos_threshold'])
        id_neg = np.where(np.sum(iou <
                                 self.params['target_args']['neg_threshold'],
                                 axis=1) == iou.shape[1])[0]
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*feature_map_shape, self.anchor_num))
        pos_equal_one[index_x, index_y, index_z] = 1

        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[
                id_pos, 3]
        if self.safe_log_target:
            eps = self.safe_log_eps
            targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
                np.maximum(gt_box_center[id_pos_gt, 3], eps) /
                np.maximum(anchors[id_pos, 3], eps))
            targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
                np.maximum(gt_box_center[id_pos_gt, 4], eps) /
                np.maximum(anchors[id_pos, 4], eps))
            targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
                np.maximum(gt_box_center[id_pos_gt, 5], eps) /
                np.maximum(anchors[id_pos, 5], eps))
        else:
            targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
                gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3])
            targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
                gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4])
            targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
                gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5])
            
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 1

        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 0

        label_dict = {'pos_equal_one': pos_equal_one,
                      'neg_equal_one': neg_equal_one,
                      'targets': targets}

        return label_dict

    @staticmethod
    def collate_batch(label_batch_list):
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = \
            torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = \
            torch.from_numpy(np.array(neg_equal_one))
        targets = \
            torch.from_numpy(np.array(targets))

        return {'targets': targets,
                'pos_equal_one': pos_equal_one,
                'neg_equal_one': neg_equal_one}

    def post_process(self, data_dict, output_dict):
        pred_box3d_list = []
        pred_box2d_list = []

        ego_protection_on = \
            self.params.get('ego_protection', {}).get('enabled', False)
        pred_box3d_owner = [] if ego_protection_on else None
        ego_box3d_list = [] if ego_protection_on else None

        for cav_id, cav_content in data_dict.items():
            if cav_id not in output_dict:
                continue
            transformation_matrix = cav_content['transformation_matrix']

            anchor_box = cav_content['anchor_box']

            prob = output_dict[cav_id]['psm']
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            reg = output_dict[cav_id]['rm']

            batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
            mask = \
                torch.gt(prob, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            if len(boxes3d) != 0:
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.params['order'])
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            transformation_matrix)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)

                if ego_protection_on:
                    pred_box3d_owner.append(torch.full(
                        (projected_boxes3d.shape[0],), cav_id == 'ego',
                        dtype=torch.bool,
                        device=projected_boxes3d.device))
                    if cav_id == 'ego':
                        ego_box3d_list.append(
                            (projected_boxes3d, scores))

        if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
            return None, None
        pred_box2d_list = torch.vstack(pred_box2d_list)
        scores = pred_box2d_list[:, -1]
        pred_box3d_tensor = torch.vstack(pred_box3d_list)

        if ego_protection_on:
            owner_is_ego = torch.cat(pred_box3d_owner)
            pred_box3d_tensor, scores = self._apply_ego_protection(
                pred_box3d_tensor, scores, ego_box3d_list, owner_is_ego)

        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           self.params['nms_thresh']
                                           )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        scores = scores[keep_index]

        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        return pred_box3d_tensor, scores

    def _apply_ego_protection(self, pred_box3d_tensor, scores,
                              ego_box3d_list, owner_is_ego):
        if pred_box3d_tensor.shape[0] == 0 or len(ego_box3d_list) == 0:
            return pred_box3d_tensor, scores

        protection_params = self.params.get('ego_protection', {})
        iou_thresh = protection_params.get('iou_thresh', 0.1)
        max_distance = protection_params.get('max_distance', None)
        min_ego_score = protection_params.get('min_ego_score', 0.0) or 0.0
        ego_boxes = torch.vstack([item[0] for item in ego_box3d_list]).cpu()
        ego_scores = torch.cat([item[1] for item in ego_box3d_list]).cpu()
        ego_scores_np = ego_scores.numpy()
        ego_polys = common_utils.convert_format(ego_boxes)

        keep = []
        for i in range(pred_box3d_tensor.shape[0]):
            if owner_is_ego[i]:
                keep.append(True)
                continue
            box_poly = common_utils.convert_format(
                pred_box3d_tensor[i:i + 1].cpu())[0]
            if max_distance is not None:
                distance = math.sqrt(box_poly.centroid.x ** 2 +
                                     box_poly.centroid.y ** 2)
                if distance > max_distance:
                    keep.append(True)
                    continue
            ious = common_utils.compute_iou(box_poly, ego_polys)
            if ious.size > 0:
                best = int(np.argmax(ious))
                suppress = bool(ious[best] > iou_thresh and
                                ego_scores_np[best] >= min_ego_score)
            else:
                suppress = False
            keep.append(not suppress)

        keep_mask = torch.tensor(keep, dtype=torch.bool,
                                 device=pred_box3d_tensor.device)
        return pred_box3d_tensor[keep_mask], scores[keep_mask]

    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        anchors_reshaped = anchors.view(-1, 7).float()
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_tensor,
                                                    pcd,
                                                    show_vis,
                                                    save_path)
