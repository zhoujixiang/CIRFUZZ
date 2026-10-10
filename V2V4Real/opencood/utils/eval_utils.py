import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.utils import box_utils
from opencood.hypes_yaml import yaml_utils


ERROR_STAT_KEYS = [
    'gt_total',
    'dt_total',
    'success_gt',
    'success_dt',
    'object_missing',
    'false_detection',
    'localization_failure_gt',
    'localization_failure_dt'
]


def init_error_stat(iou_threshs):
    error_stat = {}
    for iou_thresh in iou_threshs:
        error_stat[iou_thresh] = {
            'gt_total': 0,
            'dt_total': 0,
            'success_gt': 0,
            'success_dt': 0,
            'object_missing': 0,
            'false_detection': 0,
            'localization_failure_gt': 0,
            'localization_failure_dt': 0
        }
    return error_stat


def _init_error_summary():
    return {key: 0 for key in ERROR_STAT_KEYS}


def _range_filter_boxes(boxes, scores=None,
                        left_range=-float('inf'),
                        right_range=float('inf')):
    if boxes is None or len(boxes) == 0:
        if scores is None:
            return [], []
        return [], [], np.array([])
    polygons_origin = list(common_utils.convert_format(boxes))
    polygons = []
    keep_indices = []

    for i, polygon in enumerate(polygons_origin):
        distance = np.sqrt(polygon.centroid.x ** 2 + polygon.centroid.y ** 2)
        if left_range < distance < right_range:
            polygons.append(polygon)
            keep_indices.append(i)

    if scores is None:
        return polygons, keep_indices

    filtered_scores = scores[keep_indices] if len(keep_indices) > 0 else \
        np.array([])
    return polygons, keep_indices, filtered_scores


def _normalize_boxes_array(boxes):
    if boxes is None:
        return None
    boxes = common_utils.torch_tensor_to_numpy(boxes) \
        if isinstance(boxes, torch.Tensor) else boxes
    return boxes


def _normalize_points_array(points):
    if points is None:
        return None
    points = common_utils.torch_tensor_to_numpy(points) \
        if isinstance(points, torch.Tensor) else points
    if len(points.shape) > 2:
        points = points[0]
    return points


def filter_boxes_by_ego_lidar_visibility(boxes,
                                         ego_lidar_points,
                                         min_num_points=1):
    boxes = _normalize_boxes_array(boxes)
    ego_lidar_points = _normalize_points_array(ego_lidar_points)

    if boxes is None:
        return None, []
    if ego_lidar_points is None or ego_lidar_points.shape[0] == 0:
        empty_shape = (0,) + tuple(boxes.shape[1:])
        return np.zeros(empty_shape, dtype=boxes.dtype), []

    keep_indices = []
    points_xyz = ego_lidar_points[:, :3]

    for i, box in enumerate(boxes):
        box_points = box_utils.get_points_in_rotated_box_3d(points_xyz, box)
        if box_points.shape[0] >= min_num_points:
            keep_indices.append(i)

    filtered_boxes = boxes[keep_indices] if len(keep_indices) > 0 else \
        np.zeros((0,) + tuple(boxes.shape[1:]), dtype=boxes.dtype)
    return filtered_boxes, keep_indices


def _filter_scores_by_indices(scores, keep_indices):
    if scores is None:
        return None
    scores = common_utils.torch_tensor_to_numpy(scores) \
        if isinstance(scores, torch.Tensor) else scores
    if len(keep_indices) == 0:
        return np.zeros((0,), dtype=scores.dtype)
    return scores[keep_indices]


def get_error_type_summary(det_boxes,
                           gt_boxes,
                           iou_thresh,
                           det_scores=None,
                           ego_lidar_points=None,
                           use_ego_lidar_visibility=False,
                           left_range=-float('inf'),
                           right_range=float('inf')):
    summary = _init_error_summary()

    if gt_boxes is None:
        return summary

    gt_boxes = _normalize_boxes_array(gt_boxes)
    if use_ego_lidar_visibility:
        gt_boxes, _ = filter_boxes_by_ego_lidar_visibility(gt_boxes,
                                                           ego_lidar_points)
    gt_polygon_list, _ = _range_filter_boxes(gt_boxes,
                                             left_range=left_range,
                                             right_range=right_range)
    summary['gt_total'] = len(gt_polygon_list)

    if det_boxes is None:
        summary['object_missing'] = summary['gt_total']
        return summary

    det_boxes = _normalize_boxes_array(det_boxes)
    if use_ego_lidar_visibility:
        det_boxes, keep_indices = filter_boxes_by_ego_lidar_visibility(
            det_boxes,
            ego_lidar_points)
        det_scores = _filter_scores_by_indices(det_scores, keep_indices)
    det_polygon_list, _ = _range_filter_boxes(det_boxes,
                                              left_range=left_range,
                                              right_range=right_range)
    summary['dt_total'] = len(det_polygon_list)

    gt_total = summary['gt_total']
    dt_total = summary['dt_total']

    if gt_total == 0 and dt_total == 0:
        return summary

    if gt_total == 0:
        summary['false_detection'] = dt_total
        return summary

    if dt_total == 0:
        summary['object_missing'] = gt_total
        return summary

    iou_matrix = np.zeros((gt_total, dt_total), dtype=np.float32)
    for i, gt_polygon in enumerate(gt_polygon_list):
        iou_matrix[i] = common_utils.compute_iou(gt_polygon, det_polygon_list)

    gt_best_iou = np.max(iou_matrix, axis=1)
    dt_best_iou = np.max(iou_matrix, axis=0)

    summary['success_gt'] = int(np.sum(gt_best_iou > iou_thresh))
    summary['success_dt'] = int(np.sum(dt_best_iou > iou_thresh))
    summary['object_missing'] = int(np.sum(gt_best_iou == 0))
    summary['false_detection'] = int(np.sum(dt_best_iou == 0))
    summary['localization_failure_gt'] = int(
        np.sum((gt_best_iou > 0) & (gt_best_iou <= iou_thresh)))
    summary['localization_failure_dt'] = int(
        np.sum((dt_best_iou > 0) & (dt_best_iou <= iou_thresh)))

    return summary


def calculate_error_types(det_boxes,
                          gt_boxes,
                          result_stat,
                          iou_thresh,
                          det_scores=None,
                          ego_lidar_points=None,
                          use_ego_lidar_visibility=False,
                          left_range=-float('inf'),
                          right_range=float('inf')):
    summary = get_error_type_summary(det_boxes,
                                     gt_boxes,
                                     iou_thresh,
                                     det_scores=det_scores,
                                     ego_lidar_points=ego_lidar_points,
                                     use_ego_lidar_visibility=use_ego_lidar_visibility,
                                     left_range=left_range,
                                     right_range=right_range)
    for key, value in summary.items():
        result_stat[iou_thresh][key] += value


def voc_ap(rec, prec):
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh,
                    ego_lidar_points=None,
                    use_ego_lidar_visibility=False,
                    left_range=-float('inf'), right_range=float('inf')):
    fp = []
    tp = []
    
    if det_boxes is not None:
        det_boxes = _normalize_boxes_array(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = _normalize_boxes_array(gt_boxes)

        if use_ego_lidar_visibility:
            det_boxes, keep_indices = filter_boxes_by_ego_lidar_visibility(
                det_boxes,
                ego_lidar_points)
            det_score = _filter_scores_by_indices(det_score, keep_indices)
            gt_boxes, _ = filter_boxes_by_ego_lidar_visibility(gt_boxes,
                                                               ego_lidar_points)

        if det_boxes is None or gt_boxes is None:
            gt = _normalize_boxes_array(gt_boxes)
            gt = gt.shape[0] if gt is not None else 0
            result_stat[iou_thresh]['fp'] += fp
            result_stat[iou_thresh]['tp'] += tp
            result_stat[iou_thresh]['gt'] += gt
            return

        det_polygon_list_origin = list(common_utils.convert_format(det_boxes))
        gt_polygon_list_origin = list(common_utils.convert_format(gt_boxes))
        det_polygon_list = []
        gt_polygon_list = []
        det_score_new = []
        for i in range(len(det_polygon_list_origin)):
            det_polygon = det_polygon_list_origin[i]
            distance = np.sqrt(det_polygon.centroid.x**2 +
                               det_polygon.centroid.y**2)
            if left_range < distance < right_range:
                det_polygon_list.append(det_polygon)
                det_score_new.append(det_score[i])

        for i in range(len(gt_polygon_list_origin)):
            gt_polygon = gt_polygon_list_origin[i]
            distance = np.sqrt(gt_polygon.centroid.x**2 +
                               gt_polygon.centroid.y**2)
            if left_range < distance < right_range:
                gt_polygon_list.append(gt_polygon)

        gt = len(gt_polygon_list)
        det_score_new = np.array(det_score_new)
        score_order_descend = np.argsort(-det_score_new)

        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)
    else:
        gt_boxes = _normalize_boxes_array(gt_boxes)
        if use_ego_lidar_visibility:
            gt_boxes, _ = filter_boxes_by_ego_lidar_visibility(gt_boxes,
                                                               ego_lidar_points)
        gt = gt_boxes.shape[0]
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def calculate_ap(result_stat, iou):
    iou_5 = result_stat[iou]

    fp = iou_5['fp']
    tp = iou_5['tp']
    if not fp or not tp:
        return 0.0, 0.0, 0.0
    assert len(fp) == len(tp)

    gt_total = iou_5['gt']
    if gt_total <= 0:
        return 0.0, 0.0, 0.0

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, range=""):
    dump_dict = {}
    file_name = 'eval.yaml' if range == "" else range+'_eval.yaml'
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    dump_dict.update({'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, file_name))

    print('The range is %s, '
          'The Average Precision at IOU 0.5 is %.3f, '
          'The Average Precision at IOU 0.7 is %.3f' % (range,  ap_50, ap_70))


def summarize_error_stat(result_stat, iou_thresh):
    stats = result_stat[iou_thresh]
    gt_total = stats['gt_total']
    dt_total = stats['dt_total']

    summary = dict(stats)
    summary.update({
        'object_missing_rate': float(stats['object_missing']) / gt_total
        if gt_total > 0 else 0.0,
        'false_detection_rate': float(stats['false_detection']) / dt_total
        if dt_total > 0 else 0.0,
        'localization_failure_rate_gt':
            float(stats['localization_failure_gt']) / gt_total
            if gt_total > 0 else 0.0,
        'localization_failure_rate_dt':
            float(stats['localization_failure_dt']) / dt_total
            if dt_total > 0 else 0.0,
        'success_rate_gt': float(stats['success_gt']) / gt_total
        if gt_total > 0 else 0.0,
        'success_rate_dt': float(stats['success_dt']) / dt_total
        if dt_total > 0 else 0.0
    })
    return summary


def summarize_error_stat_dict(result_stat):
    dump_dict = {}
    for iou_thresh in result_stat:
        dump_dict['iou_%.2f' % iou_thresh] = summarize_error_stat(result_stat,
                                                                  iou_thresh)
    return dump_dict


def diff_summary_dict(summary_a, summary_b):
    diff_dict = {}
    all_keys = sorted(set(summary_a.keys()) | set(summary_b.keys()))

    for key in all_keys:
        value_a = summary_a.get(key)
        value_b = summary_b.get(key)

        if isinstance(value_a, dict) and isinstance(value_b, dict):
            diff_dict[key] = diff_summary_dict(value_a, value_b)
            continue

        if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
            diff_dict[key] = value_b - value_a
            continue

        diff_dict[key] = {'before': value_a, 'after': value_b}

    return diff_dict


def eval_final_error_results(result_stat, save_path, range=""):
    file_name = 'error_eval.yaml' if range == "" else range + '_error_eval.yaml'
    dump_dict = summarize_error_stat_dict(result_stat)

    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, file_name))

    for iou_thresh in result_stat:
        summary = dump_dict['iou_%.2f' % iou_thresh]
        print('The range is %s, IoU %.2f, missing %.3f, false detection %.3f, '
              'localization(gt) %.3f, localization(dt) %.3f' %
              (range,
               iou_thresh,
               summary['object_missing_rate'],
               summary['false_detection_rate'],
               summary['localization_failure_rate_gt'],
               summary['localization_failure_rate_dt']))
