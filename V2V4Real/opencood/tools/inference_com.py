import argparse
import copy
import os
import time

import numpy as np
import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils


IOU_THRESH_LIST = [0.5, 0.7]
RANGE_SETTINGS = [
    ('overall', -float('inf'), float('inf')),
    ('short', 0, 30),
    ('middle', 30, 50),
    ('long', 50, 100)
]
ERROR_TYPES = [
    'success_gt',
    'success_dt',
    'object_missing',
    'false_detection',
    'localization_failure_gt',
    'localization_failure_dt'
]
FRAME_ERROR_TYPES = [
    'object_missing',
    'false_detection',
    'localization_failure'
]
BOX_STATUS_KEYS = [
    'success',
    'object_missing',
    'false_detection',
    'localization_failure'
]


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--ego_only', action='store_true',
                        help='test using ego vehicle perception only')
    parser.add_argument('--compare_ego_v2x', action='store_true',
                        help='run ego-only and full v2x together and record '
                             'their error-type differences in ego range')
    parser.add_argument('--frame_ids', type=str, default='',
                        help='comma separated frame indices to visualize or '
                             'save, e.g. 0,10,25')
    parser.add_argument('--frame_start', type=int, default=0,
                        help='start frame index for visualization/save filter')
    parser.add_argument('--frame_end', type=int, default=-1,
                        help='end frame index for visualization/save filter, '
                             '-1 means no upper bound')
    parser.add_argument('--frame_stride', type=int, default=1,
                        help='frame stride for visualization/save filter')
    parser.add_argument('--log_every', type=int, default=1,
                        help='print progress every N frames')
    opt = parser.parse_args()
    return opt


def parse_frame_ids(frame_ids):
    if not frame_ids:
        return set()
    return set(int(x.strip()) for x in frame_ids.split(',') if x.strip())


def should_process_frame(frame_idx, opt):
    selected_ids = parse_frame_ids(opt.frame_ids)
    if selected_ids:
        return frame_idx in selected_ids
    if frame_idx < opt.frame_start:
        return False
    if opt.frame_end >= 0 and frame_idx > opt.frame_end:
        return False
    stride = max(opt.frame_stride, 1)
    return (frame_idx - opt.frame_start) % stride == 0


def init_eval_stats():
    result_stats = {}
    error_stats = {}

    for range_name, _, _ in RANGE_SETTINGS:
        result_stats[range_name] = {
            0.5: {'tp': [], 'fp': [], 'gt': 0},
            0.7: {'tp': [], 'fp': [], 'gt': 0}
        }
        error_stats[range_name] = eval_utils.init_error_stat(IOU_THRESH_LIST)

    return result_stats, error_stats


def get_ego_lidar_for_visibility(batch_data):
    ego_content = batch_data['ego']
    if 'ego_origin_lidar' in ego_content:
        return ego_content['ego_origin_lidar']
    return ego_content.get('origin_lidar', None)


def update_eval_stats(pred_box_tensor,
                      pred_score,
                      gt_box_tensor,
                      result_stats,
                      error_stats,
                      ego_lidar_points=None,
                      use_ego_lidar_visibility=False):
    for range_name, left_range, right_range in RANGE_SETTINGS:
        for iou_thresh in IOU_THRESH_LIST:
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stats[range_name],
                                       iou_thresh,
                                       ego_lidar_points=ego_lidar_points,
                                       use_ego_lidar_visibility=
                                       use_ego_lidar_visibility,
                                       left_range=left_range,
                                       right_range=right_range)
            eval_utils.calculate_error_types(pred_box_tensor,
                                             gt_box_tensor,
                                             error_stats[range_name],
                                             iou_thresh,
                                             det_scores=pred_score,
                                             ego_lidar_points=ego_lidar_points,
                                             use_ego_lidar_visibility=
                                             use_ego_lidar_visibility,
                                             left_range=left_range,
                                             right_range=right_range)


def run_inference_by_method(batch_data, model, dataset, fusion_method):
    if fusion_method == 'nofusion':
        return infrence_utils.inference_no_fusion(batch_data, model, dataset)
    if fusion_method == 'late':
        return infrence_utils.inference_late_fusion(batch_data, model, dataset)
    if fusion_method == 'early':
        return infrence_utils.inference_early_fusion(batch_data, model, dataset)
    if fusion_method == 'intermediate':
        return infrence_utils.inference_intermediate_fusion(batch_data,
                                                            model,
                                                            dataset)
    raise NotImplementedError('Only early, late and intermediate fusion is '
                              'supported.')


def _to_numpy_or_empty(tensor, default_shape):
    if tensor is None:
        return np.zeros(default_shape, dtype=np.float32)
    array = eval_utils.common_utils.torch_tensor_to_numpy(tensor)
    return array.astype(np.float32, copy=False)


def save_comparison_visualization(ego_pred_tensor,
                                  v2x_pred_tensor,
                                  gt_tensor,
                                  pcd,
                                  save_path,
                                  mode='constant'):
    origin_lidar = pcd
    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = eval_utils.common_utils.torch_tensor_to_numpy(pcd)
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    origin_lidar = origin_lidar.copy()

    colorized_lidar = vis_utils.color_encoding(origin_lidar[:, -1]
                                               if mode == 'intensity'
                                               else origin_lidar[:, -1],
                                               mode=mode)
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(colorized_lidar)

    elements = [o3d_pcd]
    elements += vis_utils.bbx2linset(gt_tensor, color=(0, 0, 1))
    if ego_pred_tensor is not None:
        elements += vis_utils.bbx2linset(ego_pred_tensor, color=(1, 0, 0))
    if v2x_pred_tensor is not None:
        elements += vis_utils.bbx2linset(v2x_pred_tensor, color=(0, 1, 0))

    vis_utils.save_o3d_visualization(elements, save_path)


def save_difference_case(frame_idx,
                         batch_data,
                         gt_box_tensor,
                         ego_pred_tensor,
                         v2x_pred_tensor,
                         case_payload,
                         base_save_path):
    os.makedirs(base_save_path, exist_ok=True)
    prefix = os.path.join(base_save_path, '%05d' % frame_idx)

    yaml_utils.save_yaml(case_payload, prefix + '_diff.yaml')

    pcd = batch_data['ego']['origin_lidar'][0]
    np.save(prefix + '_pcd.npy', _to_numpy_or_empty(pcd, (0, 4)))
    np.save(prefix + '_gt.npy', _to_numpy_or_empty(gt_box_tensor, (0, 8, 3)))
    np.save(prefix + '_ego_pred.npy',
            _to_numpy_or_empty(ego_pred_tensor, (0, 8, 3)))
    np.save(prefix + '_v2x_pred.npy',
            _to_numpy_or_empty(v2x_pred_tensor, (0, 8, 3)))
    if 'DISPLAY' in os.environ:
        save_comparison_visualization(ego_pred_tensor,
                                      v2x_pred_tensor,
                                      gt_box_tensor,
                                      pcd,
                                      prefix + '_compare.png')


def build_frame_meta(batch_data):
    ego_content = batch_data['ego']
    object_ids = ego_content['object_ids'] if 'object_ids' in ego_content else []
    if isinstance(object_ids, torch.Tensor):
        object_ids = object_ids.tolist()

    return {
        'scene_id': int(ego_content.get('scenario_index', -1)),
        'frame_idx_in_scene': int(ego_content.get('timestamp_index', -1)),
        'timestamp': str(ego_content.get('timestamp_key', '')),
        'ego_cav_id': int(ego_content.get('cav_id', -1)),
        'folder_name': str(ego_content.get('folder_name', '')),
        'object_ids': object_ids
    }


def classify_boxes_against_gt(pred_boxes, gt_boxes, iou_thresh):
    pred_boxes = eval_utils._normalize_boxes_array(pred_boxes)
    gt_boxes = eval_utils._normalize_boxes_array(gt_boxes)

    result = {
        'gt_status': [],
        'dt_status': [],
        'gt_best_iou': [],
        'dt_best_iou': []
    }

    gt_total = 0 if gt_boxes is None else gt_boxes.shape[0]
    dt_total = 0 if pred_boxes is None else pred_boxes.shape[0]

    if gt_total == 0:
        result['dt_status'] = ['false_detection'] * dt_total
        result['dt_best_iou'] = [0.0] * dt_total
        return result

    if dt_total == 0:
        result['gt_status'] = ['object_missing'] * gt_total
        result['gt_best_iou'] = [0.0] * gt_total
        return result

    gt_polygons = list(eval_utils.common_utils.convert_format(gt_boxes))
    dt_polygons = list(eval_utils.common_utils.convert_format(pred_boxes))
    iou_matrix = np.zeros((gt_total, dt_total), dtype=np.float32)

    for i, gt_polygon in enumerate(gt_polygons):
        iou_matrix[i] = eval_utils.common_utils.compute_iou(gt_polygon,
                                                            dt_polygons)

    gt_best_iou = np.max(iou_matrix, axis=1)
    dt_best_iou = np.max(iou_matrix, axis=0)
    result['gt_best_iou'] = gt_best_iou.tolist()
    result['dt_best_iou'] = dt_best_iou.tolist()

    for iou in gt_best_iou:
        if iou > iou_thresh:
            result['gt_status'].append('success')
        elif iou > 0:
            result['gt_status'].append('localization_failure')
        else:
            result['gt_status'].append('object_missing')

    for iou in dt_best_iou:
        if iou > iou_thresh:
            result['dt_status'].append('success')
        elif iou > 0:
            result['dt_status'].append('localization_failure')
        else:
            result['dt_status'].append('false_detection')

    return result


def summarize_frame_error_counts(error_summary):
    return {
        'object_missing': int(error_summary.get('object_missing', 0)),
        'false_detection': int(error_summary.get('false_detection', 0)),
        'localization_failure': int(
            error_summary.get('localization_failure_gt', 0) +
            error_summary.get('localization_failure_dt', 0))
    }


def build_box_level_difference(ego_pred_boxes,
                               v2x_pred_boxes,
                               gt_boxes,
                               ego_lidar_points):
    box_level = {}
    has_difference = False

    if ego_pred_boxes is None:
        ego_pred_boxes = np.zeros((0, 8, 3), dtype=np.float32)
    if v2x_pred_boxes is None:
        v2x_pred_boxes = np.zeros((0, 8, 3), dtype=np.float32)
    if gt_boxes is None:
        gt_boxes = np.zeros((0, 8, 3), dtype=np.float32)

    for range_name, left_range, right_range in RANGE_SETTINGS:
        gt_visible_boxes, gt_visible_indices = \
            eval_utils.filter_boxes_by_ego_lidar_visibility(gt_boxes,
                                                            ego_lidar_points)
        ego_visible_boxes, ego_visible_indices = \
            eval_utils.filter_boxes_by_ego_lidar_visibility(ego_pred_boxes,
                                                            ego_lidar_points)
        v2x_visible_boxes, v2x_visible_indices = \
            eval_utils.filter_boxes_by_ego_lidar_visibility(v2x_pred_boxes,
                                                            ego_lidar_points)

        gt_polygons, gt_range_indices = \
            eval_utils._range_filter_boxes(gt_visible_boxes,
                                           left_range=left_range,
                                           right_range=right_range)
        ego_polygons, ego_range_indices = \
            eval_utils._range_filter_boxes(ego_visible_boxes,
                                           left_range=left_range,
                                           right_range=right_range)
        v2x_polygons, v2x_range_indices = \
            eval_utils._range_filter_boxes(v2x_visible_boxes,
                                           left_range=left_range,
                                           right_range=right_range)

        gt_boxes_in_range = gt_visible_boxes[gt_range_indices] \
            if len(gt_range_indices) > 0 else gt_visible_boxes[:0]
        ego_boxes_in_range = ego_visible_boxes[ego_range_indices] \
            if len(ego_range_indices) > 0 else ego_visible_boxes[:0]
        v2x_boxes_in_range = v2x_visible_boxes[v2x_range_indices] \
            if len(v2x_range_indices) > 0 else v2x_visible_boxes[:0]

        gt_original_indices = [gt_visible_indices[i] for i in gt_range_indices]
        ego_original_indices = [ego_visible_indices[i] for i in ego_range_indices]
        v2x_original_indices = [v2x_visible_indices[i] for i in v2x_range_indices]

        box_level[range_name] = {}
        for iou_thresh in IOU_THRESH_LIST:
            ego_cls = classify_boxes_against_gt(ego_boxes_in_range,
                                                gt_boxes_in_range,
                                                iou_thresh)
            v2x_cls = classify_boxes_against_gt(v2x_boxes_in_range,
                                                gt_boxes_in_range,
                                                iou_thresh)
            gt_transitions = []
            transition_counter = {
                'gt': {}
            }

            for local_idx, gt_idx in enumerate(gt_original_indices):
                ego_status = ego_cls['gt_status'][local_idx] \
                    if local_idx < len(ego_cls['gt_status']) else 'object_missing'
                v2x_status = v2x_cls['gt_status'][local_idx] \
                    if local_idx < len(v2x_cls['gt_status']) else 'object_missing'
                if ego_status != v2x_status:
                    has_difference = True
                transition_name = '%s -> %s' % (ego_status, v2x_status)
                transition_counter['gt'][transition_name] = \
                    transition_counter['gt'].get(transition_name, 0) + 1
                gt_transitions.append({
                    'gt_index': int(gt_idx),
                    'ego_status': ego_status,
                    'v2x_status': v2x_status,
                    'ego_best_iou': float(ego_cls['gt_best_iou'][local_idx])
                    if local_idx < len(ego_cls['gt_best_iou']) else 0.0,
                    'v2x_best_iou': float(v2x_cls['gt_best_iou'][local_idx])
                    if local_idx < len(v2x_cls['gt_best_iou']) else 0.0
                })

            box_level[range_name]['iou_%.2f' % iou_thresh] = {
                'visible_gt_count': int(len(gt_original_indices)),
                'visible_ego_pred_count': int(len(ego_original_indices)),
                'visible_v2x_pred_count': int(len(v2x_original_indices)),
                'gt_transitions': gt_transitions,
                'transition_counter': transition_counter
            }

    return box_level, has_difference


def record_comparison_differences(frame_idx,
                                  batch_data,
                                  gt_box_tensor,
                                  ego_outputs,
                                  v2x_outputs,
                                  compare_stat,
                                  diff_case_dir,
                                  save_current_case=True):
    ego_lidar_points = get_ego_lidar_for_visibility(batch_data)
    range_differences = {}
    frame_error_counts = {}
    has_difference = False

    for range_name, left_range, right_range in RANGE_SETTINGS:
        range_differences[range_name] = {}
        frame_error_counts[range_name] = {}
        for iou_thresh in IOU_THRESH_LIST:
            ego_summary = eval_utils.get_error_type_summary(
                ego_outputs[0],
                gt_box_tensor,
                iou_thresh,
                det_scores=ego_outputs[1],
                ego_lidar_points=ego_lidar_points,
                use_ego_lidar_visibility=True,
                left_range=left_range,
                right_range=right_range)
            v2x_summary = eval_utils.get_error_type_summary(
                v2x_outputs[0],
                gt_box_tensor,
                iou_thresh,
                det_scores=v2x_outputs[1],
                ego_lidar_points=ego_lidar_points,
                use_ego_lidar_visibility=True,
                left_range=left_range,
                right_range=right_range)
            diff_summary = {}
            for key in ERROR_TYPES:
                delta = v2x_summary[key] - ego_summary[key]
                diff_summary[key] = delta
                compare_stat[range_name][iou_thresh][key] += delta
                if delta != 0:
                    has_difference = True

            frame_error_counts[range_name]['iou_%.2f' % iou_thresh] = {
                'ego_only': summarize_frame_error_counts(ego_summary),
                'v2x': summarize_frame_error_counts(v2x_summary),
                'delta_v2x_minus_ego': {
                    key: int(v2x_summary.get(key, 0) - ego_summary.get(key, 0))
                    for key in FRAME_ERROR_TYPES
                }
            }

            range_differences[range_name]['iou_%.2f' % iou_thresh] = {
                'ego_only': ego_summary,
                'v2x': v2x_summary,
                'delta_v2x_minus_ego': diff_summary
            }

    box_level_differences, box_level_has_difference = \
        build_box_level_difference(ego_outputs[0],
                                   v2x_outputs[0],
                                   gt_box_tensor,
                                   ego_lidar_points)
    has_difference = has_difference or box_level_has_difference

    if not has_difference:
        return

    compare_stat['different_frame_count'] += 1
    case_payload = {
        'frame_index': int(frame_idx),
        'meta': build_frame_meta(batch_data),
        'differences': range_differences,
        'frame_error_counts': frame_error_counts,
        'box_level_differences': box_level_differences
    }
    compare_stat['cases'].append(case_payload)
    if save_current_case:
        save_difference_case(frame_idx,
                             batch_data,
                             gt_box_tensor,
                             ego_outputs[0],
                             v2x_outputs[0],
                             case_payload,
                             diff_case_dir)


def finalize_single_run(model_dir, result_stats, error_stats):
    os.makedirs(model_dir, exist_ok=True)
    for range_name, _, _ in RANGE_SETTINGS:
        suffix = '' if range_name == 'overall' else range_name
        eval_utils.eval_final_results(result_stats[range_name],
                                      model_dir,
                                      suffix)
        eval_utils.eval_final_error_results(error_stats[range_name],
                                            model_dir,
                                            suffix)


def init_compare_stat():
    stat = {
        'frame_count': 0,
        'different_frame_count': 0,
        'cases': []
    }

    for range_name, _, _ in RANGE_SETTINGS:
        stat[range_name] = {}
        for iou_thresh in IOU_THRESH_LIST:
            stat[range_name][iou_thresh] = {key: 0 for key in ERROR_TYPES}

    return stat


def dump_compare_results(compare_stat, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        'frame_count': compare_stat['frame_count'],
        'different_frame_count': compare_stat['different_frame_count'],
        'difference_frame_rate':
            float(compare_stat['different_frame_count']) /
            compare_stat['frame_count']
            if compare_stat['frame_count'] > 0 else 0.0,
        'delta_v2x_minus_ego': {}
    }

    for range_name, _, _ in RANGE_SETTINGS:
        summary['delta_v2x_minus_ego'][range_name] = {}
        for iou_thresh in IOU_THRESH_LIST:
            summary['delta_v2x_minus_ego'][range_name][
                'iou_%.2f' % iou_thresh] = compare_stat[range_name][iou_thresh]

    yaml_utils.save_yaml(summary,
                         os.path.join(output_dir, 'compare_error_diff.yaml'))
    yaml_utils.save_yaml(compare_stat['cases'],
                         os.path.join(output_dir, 'different_cases.yaml'))


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize ' \
        'the results in single ' \
        'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)
    hypes['ego_only'] = opt.ego_only

    print('Dataset Building')
    if opt.compare_ego_v2x:
        ego_hypes = copy.deepcopy(hypes)
        ego_hypes['ego_only'] = True
        v2x_hypes = copy.deepcopy(hypes)
        v2x_hypes['ego_only'] = False

        ego_dataset = build_dataset(ego_hypes, visualize=True, train=False,
                                    isSim=opt.isSim)
        v2x_dataset = build_dataset(v2x_hypes, visualize=True, train=False,
                                    isSim=opt.isSim)
        ego_data_loader = DataLoader(ego_dataset,
                                     batch_size=1,
                                     num_workers=16,
                                     collate_fn=ego_dataset.collate_batch_test,
                                     shuffle=False,
                                     pin_memory=False,
                                     drop_last=False)
        v2x_data_loader = DataLoader(v2x_dataset,
                                     batch_size=1,
                                     num_workers=16,
                                     collate_fn=v2x_dataset.collate_batch_test,
                                     shuffle=False,
                                     pin_memory=False,
                                     drop_last=False)
        opencood_dataset = v2x_dataset
        data_loader = v2x_data_loader
    else:
        opencood_dataset = build_dataset(hypes, visualize=True, train=False,
                                         isSim=opt.isSim)
        data_loader = DataLoader(opencood_dataset,
                                 batch_size=1,
                                 num_workers=16,
                                 collate_fn=opencood_dataset.collate_batch_test,
                                 shuffle=False,
                                 pin_memory=False,
                                 drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)

    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    if opt.compare_ego_v2x:
        ego_result_stats, ego_error_stats = init_eval_stats()
        v2x_result_stats, v2x_error_stats = init_eval_stats()
        compare_stat = init_compare_stat()
        compare_output_dir = os.path.join(opt.model_dir, 'compare_ego_v2x')
        diff_case_dir = os.path.join(compare_output_dir, 'cases')
        os.makedirs(compare_output_dir, exist_ok=True)
    else:
        result_stats, error_stats = init_eval_stats()

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=bool(os.environ.get('DISPLAY')))

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().line_width = 10
        vis.get_render_option().show_coordinate_frame = True

        vis_pcd = o3d.geometry.PointCloud()
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(500):
            vis_aabbs_gt.append(o3d.geometry.TriangleMesh())
            vis_aabbs_pred.append(o3d.geometry.TriangleMesh())
        sequence_initialized = False

    data_iterable = zip(ego_data_loader, v2x_data_loader) if \
        opt.compare_ego_v2x else enumerate(data_loader)

    for iter_item in data_iterable:
        if opt.compare_ego_v2x:
            ego_batch_data, v2x_batch_data = iter_item
            i = compare_stat['frame_count']
        else:
            i, batch_data = iter_item
        frame_selected = should_process_frame(i, opt)
        if i % max(opt.log_every, 1) == 0:
            print(i)
        with torch.no_grad():
            torch.cuda.synchronize()
            if opt.compare_ego_v2x:
                ego_batch_data = train_utils.to_device(ego_batch_data, device)
                v2x_batch_data = train_utils.to_device(v2x_batch_data, device)
                ego_fusion_method = 'nofusion' \
                    if opt.fusion_method == 'late' else opt.fusion_method
                ego_outputs = run_inference_by_method(ego_batch_data,
                                                      model,
                                                      ego_dataset,
                                                      ego_fusion_method)
                v2x_outputs = run_inference_by_method(v2x_batch_data,
                                                      model,
                                                      v2x_dataset,
                                                      opt.fusion_method)

                update_eval_stats(ego_outputs[0],
                                  ego_outputs[1],
                                  ego_outputs[2],
                                  ego_result_stats,
                                  ego_error_stats,
                                  ego_lidar_points=
                                  get_ego_lidar_for_visibility(ego_batch_data),
                                  use_ego_lidar_visibility=True)
                update_eval_stats(v2x_outputs[0],
                                  v2x_outputs[1],
                                  ego_outputs[2],
                                  v2x_result_stats,
                                  v2x_error_stats,
                                  ego_lidar_points=
                                  get_ego_lidar_for_visibility(v2x_batch_data),
                                  use_ego_lidar_visibility=True)
                compare_stat['frame_count'] += 1
                record_comparison_differences(i,
                                              v2x_batch_data,
                                              ego_outputs[2],
                                              ego_outputs,
                                              v2x_outputs,
                                              compare_stat,
                                              diff_case_dir,
                                              save_current_case=frame_selected)

                pred_box_tensor = v2x_outputs[0]
                gt_box_tensor = v2x_outputs[2]
                batch_data = v2x_batch_data
            else:
                batch_data = train_utils.to_device(batch_data, device)
                fusion_method = opt.fusion_method
                if opt.ego_only and fusion_method == 'late':
                    fusion_method = 'nofusion'

                pred_box_tensor, pred_score, gt_box_tensor = \
                    run_inference_by_method(batch_data,
                                            model,
                                            opencood_dataset,
                                            fusion_method)
                update_eval_stats(pred_box_tensor,
                                  pred_score,
                                  gt_box_tensor,
                                  result_stats,
                                  error_stats)

            if opt.save_npy and not opt.compare_ego_v2x and frame_selected:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                infrence_utils.save_prediction_gt(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  i,
                                                  npy_save_path)

            if (opt.show_vis or opt.save_vis) and frame_selected:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence and frame_selected:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'][0],
                        vis_pcd,
                        mode='constant'
                    )
                if not sequence_initialized:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')
                    sequence_initialized = True

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    if opt.compare_ego_v2x:
        ego_output_dir = os.path.join(compare_output_dir, 'ego_only')
        v2x_output_dir = os.path.join(compare_output_dir, 'v2x')
        finalize_single_run(ego_output_dir, ego_result_stats, ego_error_stats)
        finalize_single_run(v2x_output_dir, v2x_result_stats, v2x_error_stats)
        ego_summary = {
            range_name: eval_utils.summarize_error_stat_dict(
                ego_error_stats[range_name])
            for range_name, _, _ in RANGE_SETTINGS
        }
        v2x_summary = {
            range_name: eval_utils.summarize_error_stat_dict(
                v2x_error_stats[range_name])
            for range_name, _, _ in RANGE_SETTINGS
        }
        yaml_utils.save_yaml({
            'ego_only': ego_summary,
            'v2x': v2x_summary,
            'delta_v2x_minus_ego': eval_utils.diff_summary_dict(ego_summary,
                                                                v2x_summary)
        }, os.path.join(compare_output_dir, 'error_summary_compare.yaml'))
        dump_compare_results(compare_stat, compare_output_dir)
    else:
        finalize_single_run(opt.model_dir, result_stats, error_stats)
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
