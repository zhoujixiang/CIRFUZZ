import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils, infrence_utils
from opencood.utils import box_utils, common_utils, eval_utils

IOU_THRESH_LIST = [0.5, 0.7]
RANGE_SETTINGS = [
    ('overall', -float('inf'), float('inf')),
    ('short', 0, 30),
    ('middle', 30, 50),
    ('long', 50, 100)
]

EGO_RANGE_ASSOC_DIST = 2.0


def test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--fusion_method', type=str, default='late')
    parser.add_argument('--frame_start', type=int, default=0)
    parser.add_argument('--frame_end', type=int, default=-1)
    parser.add_argument('--frame_stride', type=int, default=1)
    parser.add_argument('--max_frames', type=int, default=-1)
    parser.add_argument('--isSim', action='store_true')
    parser.add_argument('--no_error_types', action='store_true',
                        help='skip the expensive per-frame error-type '
                             'matching (AP-only run, much faster)')
    opt = parser.parse_args()
    return opt


def run_inference(batch_data, model, dataset, fusion_method):
    if fusion_method == 'nofusion':
        return infrence_utils.inference_no_fusion(batch_data, model, dataset)
    if fusion_method == 'late':
        return infrence_utils.inference_late_fusion(batch_data, model, dataset)
    if fusion_method == 'early':
        return infrence_utils.inference_early_fusion(batch_data, model, dataset)
    if fusion_method == 'intermediate':
        return infrence_utils.inference_intermediate_fusion(batch_data,
                                                            model, dataset)
    raise NotImplementedError(fusion_method)


def to_numpy_or_empty(tensor, default_shape):
    if tensor is None:
        return np.zeros(default_shape, dtype=np.float32)
    return common_utils.torch_tensor_to_numpy(tensor).astype(np.float32,
                                                             copy=False)


def box_center_to_corners_np(boxes, order='hwl'):
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    return common_utils.torch_tensor_to_numpy(
        box_utils.boxes_to_corners_3d(boxes, order))


def get_ego_origin_lidar(batch_data):
    ego_content = batch_data['ego']
    if 'ego_origin_lidar' in ego_content:
        return ego_content['ego_origin_lidar']
    return ego_content.get('origin_lidar', None)


def _normalize_boxes_array(boxes):
    if boxes is None:
        return np.zeros((0, 8, 3), dtype=np.float32)
    boxes = common_utils.torch_tensor_to_numpy(boxes)
    if len(boxes.shape) == 2:
        return box_utils.boxes_to_corners_3d(boxes, 'hwl')
    return boxes.astype(np.float32, copy=False)


class BoxOwnership:

    def __init__(self, post_processor, nms_thresh):
        self.pp = post_processor
        self.nms_thresh = nms_thresh

    def decode_per_cav(self, data_dict, output_dict):
        ego_boxes, ego_scores = [], []
        non_ego_boxes, non_ego_scores = [], []
        all_proj, all_scores = [], []

        for cav_id, cav_content in data_dict.items():
            if cav_id not in output_dict:
                continue
            transformation_matrix = cav_content['transformation_matrix']
            anchor_box = cav_content['anchor_box']

            prob = output_dict[cav_id]['psm']
            prob = torch.nn.functional.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            reg = output_dict[cav_id]['rm']
            batch_box3d = self.pp.delta_to_boxes3d(reg, anchor_box)
            mask = torch.gt(prob, self.pp.params['target_args'][
                'score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            if len(boxes3d) == 0:
                continue
            boxes3d_corner = box_utils.boxes_to_corners_3d(
                boxes3d, order=self.pp.params['order'])
            projected_boxes3d = box_utils.project_box3d(
                boxes3d_corner, transformation_matrix)
            projected_boxes3d_np = common_utils.torch_tensor_to_numpy(
                projected_boxes3d).astype(np.float32)
            scores_np = common_utils.torch_tensor_to_numpy(scores).astype(
                np.float32)

            all_proj.append(projected_boxes3d_np)
            all_scores.append(scores_np)

            if cav_id == 'ego':
                ego_boxes.append(projected_boxes3d_np)
                ego_scores.append(scores_np)
            else:
                non_ego_boxes.append(projected_boxes3d_np)
                non_ego_scores.append(scores_np)

        def _stack(parts):
            if len(parts) == 0:
                return np.zeros((0, 8, 3), dtype=np.float32)
            return np.vstack(parts).astype(np.float32)

        return (_stack(ego_boxes),
                np.concatenate(ego_scores) if ego_scores else
                np.zeros((0,), dtype=np.float32),
                _stack(non_ego_boxes),
                np.concatenate(non_ego_scores) if non_ego_scores else
                np.zeros((0,), dtype=np.float32),
                _stack(all_proj),
                np.concatenate(all_scores) if all_scores else
                np.zeros((0,), dtype=np.float32))

    def merged_nms(self, boxes, scores):
        if len(boxes) == 0:
            return np.zeros((0, 8, 3), dtype=np.float32), \
                np.zeros((0,), dtype=np.float32)
        keep = box_utils.nms_rotated(
            torch.from_numpy(boxes.reshape(len(boxes), 8, 3)),
            torch.from_numpy(scores),
            self.nms_thresh)
        return boxes[keep], scores[keep]


def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)

    ego_hypes = dict(hypes)
    ego_hypes['ego_only'] = True
    v2x_hypes = dict(hypes)
    v2x_hypes['ego_only'] = False

    ego_dataset = build_dataset(ego_hypes, visualize=True, train=False,
                                isSim=opt.isSim)
    v2x_dataset = build_dataset(v2x_hypes, visualize=True, train=False,
                                isSim=opt.isSim)

    ego_loader = DataLoader(ego_dataset, batch_size=1, num_workers=16,
                            collate_fn=ego_dataset.collate_batch_test,
                            shuffle=False, pin_memory=False, drop_last=False)
    v2x_loader = DataLoader(v2x_dataset, batch_size=1, num_workers=16,
                            collate_fn=v2x_dataset.collate_batch_test,
                            shuffle=False, pin_memory=False, drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    ownership = BoxOwnership(v2x_dataset.post_processor,
                             hypes['postprocess']['nms_thresh'])

    merged_in_ego_range = {r: 0 for r, _, _ in RANGE_SETTINGS}
    ego_driven_in_ego_range = {r: 0 for r, _, _ in RANGE_SETTINGS}
    non_ego_in_ego_range = {r: 0 for r, _, _ in RANGE_SETTINGS}

    ap_result_merged = {r: {iou: {'tp': [], 'fp': [], 'gt': 0}
                            for iou in IOU_THRESH_LIST}
                        for r, _, _ in RANGE_SETTINGS}
    ap_result_protected = {r: {iou: {'tp': [], 'fp': [], 'gt': 0}
                               for iou in IOU_THRESH_LIST}
                           for r, _, _ in RANGE_SETTINGS}
    err_merged = {r: eval_utils.init_error_stat(IOU_THRESH_LIST)
                  for r, _, _ in RANGE_SETTINGS}
    err_protected = {r: eval_utils.init_error_stat(IOU_THRESH_LIST)
                     for r, _, _ in RANGE_SETTINGS}

    total_frames = len(v2x_dataset)
    frame_count = 0

    for (ego_batch_data, v2x_batch_data) in zip(ego_loader, v2x_loader):
        i = frame_count
        if i < opt.frame_start:
            frame_count += 1
            continue
        if opt.frame_end >= 0 and i > opt.frame_end:
            break
        if (i - opt.frame_start) % opt.frame_stride != 0:
            frame_count += 1
            continue
        if opt.max_frames > 0 and frame_count >= opt.max_frames:
            break
        if i % max(opt.frame_stride, 1) == 0:
            print('frame %d / %d' % (i, total_frames))

        with torch.no_grad():
            torch.cuda.synchronize()
            ego_batch_data = train_utils.to_device(ego_batch_data, device)
            v2x_batch_data = train_utils.to_device(v2x_batch_data, device)

            ego_outputs = run_inference(ego_batch_data, model, ego_dataset,
                                        'nofusion')

            v2x_output_dict = {
                k: model(cav_content)
                for k, cav_content in v2x_batch_data.items()
                if k in v2x_batch_data
            }
            v2x_outputs = v2x_dataset.post_process(v2x_batch_data,
                                                   v2x_output_dict)

            ego_pred = to_numpy_or_empty(ego_outputs[0], (0, 8, 3))
            ego_score = to_numpy_or_empty(ego_outputs[1], (0,))
            v2x_pred = to_numpy_or_empty(v2x_outputs[0], (0, 8, 3))
            v2x_score = to_numpy_or_empty(v2x_outputs[1], (0,))
            gt = to_numpy_or_empty(ego_outputs[2], (0, 8, 3))

            ego_corners, ego_sc, non_ego_corners, non_ego_sc, \
                all_corners, all_scores = ownership.decode_per_cav(
                    v2x_batch_data, v2x_output_dict)

            ego_lidar_points = get_ego_origin_lidar(v2x_batch_data)
            ego_lidar_points = common_utils.torch_tensor_to_numpy(
                ego_lidar_points) if ego_lidar_points is not None else None
            if ego_lidar_points is not None and len(ego_lidar_points.shape) > 2:
                ego_lidar_points = ego_lidar_points[0]

            gt_vis, _ = eval_utils.filter_boxes_by_ego_lidar_visibility(
                gt, ego_lidar_points)

            outside_boxes = np.zeros((0, 8, 3), dtype=np.float32)
            outside_scores = np.zeros((0,), dtype=np.float32)
            if len(v2x_pred) > 0:
                if len(gt_vis) > 0:
                    v2x_cents = v2x_pred[:, :, :2].mean(axis=1)
                    gt_cents = gt_vis[:, :, :2].mean(axis=1)
                    dist = np.linalg.norm(v2x_cents[:, None, :] -
                                          gt_cents[None, :, :], axis=2)
                    near_vis_gt = dist.min(axis=1) < EGO_RANGE_ASSOC_DIST
                else:
                    near_vis_gt = np.zeros(len(v2x_pred), dtype=bool)
                outside_boxes = v2x_pred[~near_vis_gt]
                outside_scores = v2x_score[~near_vis_gt]

            if len(outside_boxes) > 0 and len(ego_pred) > 0:
                cand_boxes = np.vstack([outside_boxes, ego_pred])
                cand_scores = np.concatenate([outside_scores, ego_score])
            elif len(outside_boxes) > 0:
                cand_boxes = outside_boxes
                cand_scores = outside_scores
            elif len(ego_pred) > 0:
                cand_boxes = ego_pred
                cand_scores = ego_score
            else:
                cand_boxes = np.zeros((0, 8, 3), dtype=np.float32)
                cand_scores = np.zeros((0,), dtype=np.float32)

            if len(cand_boxes) > 0:
                keep = box_utils.nms_rotated(
                    torch.from_numpy(cand_boxes.reshape(len(cand_boxes), 8, 3)),
                    torch.from_numpy(cand_scores),
                    hypes['postprocess']['nms_thresh'])
                protected_boxes = cand_boxes[keep]
                protected_scores = cand_scores[keep]
            else:
                protected_boxes = np.zeros((0, 8, 3), dtype=np.float32)
                protected_scores = np.zeros((0,), dtype=np.float32)

            merged_boxes_t = v2x_outputs[0]
            merged_scores_t = v2x_outputs[1]
            protected_boxes_t = torch.from_numpy(protected_boxes).float()
            protected_scores_t = torch.from_numpy(protected_scores).float()
            if merged_boxes_t.is_cuda:
                protected_boxes_t = protected_boxes_t.cuda()
                protected_scores_t = protected_scores_t.cuda()

            for range_name, left_range, right_range in RANGE_SETTINGS:
                for iou_thresh in IOU_THRESH_LIST:
                    eval_utils.caluclate_tp_fp(
                        merged_boxes_t, merged_scores_t, gt,
                        ap_result_merged[range_name], iou_thresh,
                        left_range=left_range, right_range=right_range)
                    eval_utils.caluclate_tp_fp(
                        protected_boxes_t, protected_scores_t, gt,
                        ap_result_protected[range_name], iou_thresh,
                        left_range=left_range, right_range=right_range)
                    if not opt.no_error_types:
                        eval_utils.calculate_error_types(
                            merged_boxes_t, gt, err_merged[range_name],
                            iou_thresh, det_scores=merged_scores_t,
                            ego_lidar_points=ego_lidar_points,
                            use_ego_lidar_visibility=True,
                            left_range=left_range, right_range=right_range)
                        eval_utils.calculate_error_types(
                            protected_boxes_t, gt, err_protected[range_name],
                            iou_thresh, det_scores=protected_scores_t,
                            ego_lidar_points=ego_lidar_points,
                            use_ego_lidar_visibility=True,
                            left_range=left_range, right_range=right_range)

            if len(v2x_pred) > 0 and len(gt_vis) > 0:
                v2x_cents = v2x_pred[:, :, :2].mean(axis=1)
                gt_cents = gt_vis[:, :, :2].mean(axis=1)
                dist = np.linalg.norm(v2x_cents[:, None, :] -
                                      gt_cents[None, :, :], axis=2)
                near_vis_gt = dist.min(axis=1) < EGO_RANGE_ASSOC_DIST

                if len(non_ego_corners) > 0:
                    ne_cents = non_ego_corners[:, :, :2].mean(axis=1)
                    ne_dist = np.linalg.norm(ne_cents[:, None, :] -
                                             gt_cents[None, :, :], axis=2)
                    ne_near_gt = ne_dist.min(axis=1) < EGO_RANGE_ASSOC_DIST

                merged_in = v2x_pred[near_vis_gt]
                if len(merged_in) > 0:
                    merged_radii = np.linalg.norm(
                        merged_in[:, :, :2].mean(axis=1), axis=1)
                    for range_name, left_range, right_range in RANGE_SETTINGS:
                        in_range = (merged_radii > left_range) & \
                                   (merged_radii < right_range)
                        cnt = int(np.sum(in_range))
                        if cnt == 0:
                            continue
                        merged_in_ego_range[range_name] += cnt
                        for b in merged_in[in_range]:
                            if len(ego_pred) > 0:
                                eg_polys = list(
                                    common_utils.convert_format(ego_pred))
                                b_poly = common_utils.convert_format(
                                    b[None])[0]
                                ious = common_utils.compute_iou(
                                    b_poly, eg_polys)
                                if np.any(ious > 0.5):
                                    ego_driven_in_ego_range[range_name] += 1

                if len(non_ego_corners) > 0:
                    ne_in = non_ego_corners[ne_near_gt]
                    if len(ne_in) > 0:
                        ne_radii = np.linalg.norm(
                            ne_in[:, :, :2].mean(axis=1), axis=1)
                        for range_name, left_range, right_range in \
                                RANGE_SETTINGS:
                            non_ego_in_ego_range[range_name] += int(
                                np.sum((ne_radii > left_range) &
                                       (ne_radii < right_range)))

        frame_count += 1

    print('Finalizing')
    def _safe_ap(ap_result, iou_thresh):
        if ap_result[iou_thresh]['gt'] == 0:
            return (0.0, [], [])
        return eval_utils.calculate_ap(ap_result, iou_thresh)

    ap_merged = {}
    ap_protected = {}
    for range_name, _, _ in RANGE_SETTINGS:
        ap_merged[range_name] = {}
        ap_protected[range_name] = {}
        for iou_thresh in IOU_THRESH_LIST:
            ap_merged[range_name][iou_thresh] = \
                _safe_ap(ap_result_merged[range_name], iou_thresh)
            ap_protected[range_name][iou_thresh] = \
                _safe_ap(ap_result_protected[range_name], iou_thresh)

    err_merged_summary = None
    err_protected_summary = None
    if not opt.no_error_types:
        err_merged_summary = {
            r: eval_utils.summarize_error_stat_dict(err_merged[r])
            for r, _, _ in RANGE_SETTINGS}
        err_protected_summary = {
            r: eval_utils.summarize_error_stat_dict(err_protected[r])
            for r, _, _ in RANGE_SETTINGS}

    out = {
        'model_dir': opt.model_dir,
        'num_frames_processed': frame_count,
        'ego_range_assoc_dist': EGO_RANGE_ASSOC_DIST,
        'merged_in_ego_range': merged_in_ego_range,
        'ego_driven_in_ego_range': ego_driven_in_ego_range,
        'non_ego_in_ego_range': non_ego_in_ego_range,
        'ap_merged': {r: {str(i): v[0] for i, v in d.items()}
                      for r, d in ap_merged.items()},
        'ap_protected_plan1': {r: {str(i): v[0] for i, v in d.items()}
                               for r, d in ap_protected.items()},
        'error_merged': err_merged_summary,
        'error_protected_plan1': err_protected_summary,
    }
    out['no_error_types'] = opt.no_error_types

    save_path = os.path.join(opt.model_dir,
                             'ego_range_protection_diag.yaml')
    yaml_utils.save_yaml(out, save_path)
    print('Saved to %s' % save_path)
    for r, _, _ in RANGE_SETTINGS:
        print('range %s' % r)
        for iou in IOU_THRESH_LIST:
            ap_m = ap_merged[r][iou][0]
            ap_p = ap_protected[r][iou][0]
            print('  iou %s: merged AP=%.4f  protected AP=%.4f' % (
                iou, ap_m, ap_p))
            if err_merged_summary is not None:
                em = err_merged_summary[r]['iou_%.2f' % iou]
                ep = err_protected_summary[r]['iou_%.2f' % iou]
                print('    merged  : success_gt=%d object_missing=%d '
                      'loc_fail_gt=%d false_det=%d' % (
                          em['success_gt'], em['object_missing'],
                          em['localization_failure_gt'],
                          em['false_detection']))
                print('    protect : success_gt=%d object_missing=%d '
                      'loc_fail_gt=%d false_det=%d' % (
                          ep['success_gt'], ep['object_missing'],
                          ep['localization_failure_gt'],
                          ep['false_detection']))
        print('  merged boxes in ego range: %d (ego-driven %d, '
              'non-ego boxes in range %d)' % (
                  merged_in_ego_range[r], ego_driven_in_ego_range[r],
                  non_ego_in_ego_range[r]))


if __name__ == '__main__':
    main()
