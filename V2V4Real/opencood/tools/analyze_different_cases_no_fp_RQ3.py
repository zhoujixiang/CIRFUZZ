
import argparse
import json
from collections import Counter
from pathlib import Path

import yaml



DEFAULT_MODEL_NAMES = [
    'attfuse_ego_8_26',
    'coalign_ego_8_09',
    'CoBEVT_ego_8_30',
    'cost_ego',
    'early_fusion_ego',
    'Fcooper_ego',
    'pointpillar_late_ego',
    'second_early_ego',
    'second_late_ego'
]

ERROR_KEYS = [
    'gt_total',
    'dt_total',
    'success_gt',
    'success_dt',
    'object_missing',
    'false_detection',
    'localization_failure_gt',
    'localization_failure_dt',
]

RANGE_CHOICES = ['overall', 'short', 'middle', 'long']
IOU_CHOICES = ['0.50', '0.70']


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Analyze compare_ego_v2x/different_cases.yaml and summarize '
            'frames where ego-only detects objects but v2x fails on them.'
        )
    )
    parser.add_argument(
        'model_dirs',
        nargs='*',
        help=(
            'Model directories to analyze. If omitted, use the 9 default '
            'model directories under --model-root.'
        ),
    )
    parser.add_argument(
        '--model-root',
        type=Path,
        default=Path(__file__).resolve().parents[2] / 'model',
        help='Root directory used with the default model list.',
    )
    parser.add_argument(
        '--range-name',
        choices=RANGE_CHOICES,
        default='overall',
        help='Distance range to analyze.',
    )
    parser.add_argument(
        '--iou',
        choices=IOU_CHOICES,
        default='0.70',
        help='IoU threshold to analyze.',
    )
    parser.add_argument(
        '--source-status',
        default='success',
        help='Source ego status used to select target objects.',
    )
    parser.add_argument(
        '--target-statuses',
        default='object_missing',
        help=(
            'Comma separated v2x statuses to count as failure for those '
            'ego-detected GT objects, e.g. object_missing or '
            'object_missing,localization_failure.'
        ),
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        default=None,
        help='Optional path to save the full summary as JSON.',
    )
    parser.add_argument(
        '--show-frame-indices',
        action='store_true',
        help='Print selected frame indices for each model.',
    )
    return parser


def get_yaml_loader():
    return getattr(yaml, 'CSafeLoader', yaml.SafeLoader)


def load_yaml_list(yaml_path):
    with yaml_path.open('r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=get_yaml_loader())
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f'{yaml_path} is expected to contain a YAML list.')
    return data


def resolve_model_dirs(args):
    if args.model_dirs:
        return [Path(p).resolve() for p in args.model_dirs]
    return [(args.model_root / name).resolve() for name in DEFAULT_MODEL_NAMES]


def normalize_statuses(status_text):
    statuses = [item.strip() for item in status_text.split(',') if item.strip()]
    if not statuses:
        raise ValueError('At least one target status must be provided.')
    return statuses


def make_stats_template():
    return {key: 0 for key in ERROR_KEYS}


def add_error_stats(accumulator, stats):
    for key in ERROR_KEYS:
        accumulator[key] += int(stats.get(key, 0))


def finalize_error_stats(stats):
    output = dict(stats)
    output['localization_failure'] = (
        int(output.get('localization_failure_gt', 0)) +
        int(output.get('localization_failure_dt', 0))
    )
    return output


def get_case_slice(case_data, range_name, iou_key):
    differences = (
        case_data.get('differences', {})
        .get(range_name, {})
        .get(iou_key)
    )
    box_level = (
        case_data.get('box_level_differences', {})
        .get(range_name, {})
        .get(iou_key)
    )
    if differences is None or box_level is None:
        frame_index = case_data.get('frame_index', 'unknown')
        raise KeyError(
            f'Frame {frame_index} is missing {range_name}/{iou_key} details.'
        )
    return differences, box_level


def analyze_model(model_dir, range_name, iou_key, source_status, target_statuses):
    different_cases_path = model_dir / 'compare_ego_v2x' / 'different_cases.yaml'
    cases = load_yaml_list(different_cases_path)

    selected_frames = []
    transition_counter = Counter()
    total_transition_counter = Counter()
    mode_totals = {
        'ego_only': make_stats_template(),
        'v2x': make_stats_template(),
        'delta_v2x_minus_ego': {key: 0 for key in ERROR_KEYS if key not in {'gt_total', 'dt_total'}},
    }
    selected_object_count = 0

    for case_data in cases:
        differences, box_level = get_case_slice(case_data, range_name, iou_key)
        gt_transitions = box_level.get('gt_transitions', [])

        matched_transitions = []
        frame_transition_counter = Counter()
        for transition in gt_transitions:
            transition_name = (
                f"{transition.get('ego_status', 'unknown')} -> "
                f"{transition.get('v2x_status', 'unknown')}"
            )
            total_transition_counter[transition_name] += 1

            if transition.get('ego_status') != source_status:
                continue
            if transition.get('v2x_status') not in target_statuses:
                continue

            matched_transitions.append(transition)
            frame_transition_counter[transition_name] += 1

        if not matched_transitions:
            continue

        selected_object_count += len(matched_transitions)
        transition_counter.update(frame_transition_counter)
        selected_frames.append({
            'frame_index': int(case_data.get('frame_index', -1)),
            'match_count': len(matched_transitions),
            'matched_transitions': dict(frame_transition_counter),
        })

        add_error_stats(mode_totals['ego_only'], differences.get('ego_only', {}))
        add_error_stats(mode_totals['v2x'], differences.get('v2x', {}))
        for key, value in differences.get('delta_v2x_minus_ego', {}).items():
            if key in mode_totals['delta_v2x_minus_ego']:
                mode_totals['delta_v2x_minus_ego'][key] += int(value)

    ego_stats = finalize_error_stats(mode_totals['ego_only'])
    v2x_stats = finalize_error_stats(mode_totals['v2x'])
    delta_stats = finalize_error_stats(mode_totals['delta_v2x_minus_ego'])

    frame_indices = [item['frame_index'] for item in selected_frames]
    return {
        'model_name': model_dir.name,
        'model_dir': str(model_dir),
        'different_cases_path': str(different_cases_path),
        'different_case_count': len(cases),
        'selected_frame_count': len(selected_frames),
        'selected_frame_indices': frame_indices,
        'selected_object_count': selected_object_count,
        'selection': {
            'range_name': range_name,
            'iou_key': iou_key,
            'source_status': source_status,
            'target_statuses': target_statuses,
        },
        'matched_transition_counts': dict(sorted(transition_counter.items())),
        'all_transition_counts': dict(sorted(total_transition_counter.items())),
        'selected_frames': selected_frames,
        'selected_frame_totals': {
            'ego_only': ego_stats,
            'v2x': v2x_stats,
            'delta_v2x_minus_ego': delta_stats,
        },
        'success_gt_change_in_selected_frames': (
            int(v2x_stats['success_gt']) - int(ego_stats['success_gt'])
        ),
        'success_dt_change_in_selected_frames': (
            int(v2x_stats['success_dt']) - int(ego_stats['success_dt'])
        ),
    }


def aggregate_results(per_model_results):
    aggregate = {
        'model_count': 0,
        'selected_frame_count': 0,
        'selected_object_count': 0,
        'different_case_count': 0,
        'selected_frame_totals': {
            'ego_only': make_stats_template(),
            'v2x': make_stats_template(),
            'delta_v2x_minus_ego': {key: 0 for key in ERROR_KEYS if key not in {'gt_total', 'dt_total'}},
        },
        'matched_transition_counts': Counter(),
    }

    for result in per_model_results:
        aggregate['model_count'] += 1
        aggregate['selected_frame_count'] += result['selected_frame_count']
        aggregate['selected_object_count'] += result['selected_object_count']
        aggregate['different_case_count'] += result['different_case_count']
        add_error_stats(
            aggregate['selected_frame_totals']['ego_only'],
            result['selected_frame_totals']['ego_only'],
        )
        add_error_stats(
            aggregate['selected_frame_totals']['v2x'],
            result['selected_frame_totals']['v2x'],
        )
        for key, value in result['selected_frame_totals']['delta_v2x_minus_ego'].items():
            if key in aggregate['selected_frame_totals']['delta_v2x_minus_ego']:
                aggregate['selected_frame_totals']['delta_v2x_minus_ego'][key] += int(value)
        aggregate['matched_transition_counts'].update(
            result['matched_transition_counts']
        )

    aggregate['selected_frame_totals']['ego_only'] = finalize_error_stats(
        aggregate['selected_frame_totals']['ego_only']
    )
    aggregate['selected_frame_totals']['v2x'] = finalize_error_stats(
        aggregate['selected_frame_totals']['v2x']
    )
    aggregate['selected_frame_totals']['delta_v2x_minus_ego'] = finalize_error_stats(
        aggregate['selected_frame_totals']['delta_v2x_minus_ego']
    )
    aggregate['matched_transition_counts'] = dict(
        sorted(aggregate['matched_transition_counts'].items())
    )
    aggregate['success_gt_change_in_selected_frames'] = (
        int(aggregate['selected_frame_totals']['v2x']['success_gt']) -
        int(aggregate['selected_frame_totals']['ego_only']['success_gt'])
    )
    aggregate['success_dt_change_in_selected_frames'] = (
        int(aggregate['selected_frame_totals']['v2x']['success_dt']) -
        int(aggregate['selected_frame_totals']['ego_only']['success_dt'])
    )
    return aggregate


def format_mode_stats(prefix, stats):
    return (
        f"{prefix}: gt_total={stats['gt_total']}, dt_total={stats['dt_total']}, "
        f"success_gt={stats['success_gt']}, success_dt={stats['success_dt']}, "
        f"object_missing={stats['object_missing']}, "
        f"false_detection={stats['false_detection']}, "
        f"localization_failure={stats['localization_failure']} "
        f"(gt={stats['localization_failure_gt']}, "
        f"dt={stats['localization_failure_dt']})"
    )


def print_result(result, show_frame_indices=False):
    print(f"[{result['model_name']}]")
    print(
        '  '
        f"selected_frames={result['selected_frame_count']}, "
        f"selected_objects={result['selected_object_count']}, "
        f"different_cases={result['different_case_count']}"
    )
    print(
        '  '
        f"success_gt_change={result['success_gt_change_in_selected_frames']}, "
        f"success_dt_change={result['success_dt_change_in_selected_frames']}"
    )
    print(
        '  ' +
        format_mode_stats('ego_only', result['selected_frame_totals']['ego_only'])
    )
    print(
        '  ' +
        format_mode_stats('v2x', result['selected_frame_totals']['v2x'])
    )
    delta_stats = result['selected_frame_totals']['delta_v2x_minus_ego']
    print(
        '  '
        f"delta_v2x_minus_ego: success_gt={delta_stats['success_gt']}, "
        f"success_dt={delta_stats['success_dt']}, "
        f"object_missing={delta_stats['object_missing']}, "
        f"false_detection={delta_stats['false_detection']}, "
        f"localization_failure={delta_stats['localization_failure']} "
        f"(gt={delta_stats['localization_failure_gt']}, "
        f"dt={delta_stats['localization_failure_dt']})"
    )
    print(f"  matched_transitions={result['matched_transition_counts']}")
    if show_frame_indices:
        print(f"  frame_indices={result['selected_frame_indices']}")
    print()


def main():
    args = build_parser().parse_args()
    model_dirs = resolve_model_dirs(args)
    target_statuses = normalize_statuses(args.target_statuses)
    iou_key = f'iou_{args.iou}'

    print(
        'Selection: '
        f"range={args.range_name}, iou={iou_key}, "
        f"ego_status={args.source_status}, "
        f"v2x_statuses={target_statuses}"
    )
    print()

    results = []
    missing = []
    for model_dir in model_dirs:
        different_cases_path = model_dir / 'compare_ego_v2x' / 'different_cases.yaml'
        if not different_cases_path.exists():
            missing.append(str(different_cases_path))
            continue
        results.append(
            analyze_model(
                model_dir=model_dir,
                range_name=args.range_name,
                iou_key=iou_key,
                source_status=args.source_status,
                target_statuses=target_statuses,
            )
        )

    for result in results:
        print_result(result, show_frame_indices=args.show_frame_indices)

    aggregate = aggregate_results(results)
    print('[aggregate]')
    print(
        '  '
        f"models={aggregate['model_count']}, "
        f"selected_frames={aggregate['selected_frame_count']}, "
        f"selected_objects={aggregate['selected_object_count']}, "
        f"different_cases={aggregate['different_case_count']}"
    )
    print(
        '  '
        f"success_gt_change={aggregate['success_gt_change_in_selected_frames']}, "
        f"success_dt_change={aggregate['success_dt_change_in_selected_frames']}"
    )
    print(
        '  ' +
        format_mode_stats('ego_only', aggregate['selected_frame_totals']['ego_only'])
    )
    print(
        '  ' +
        format_mode_stats('v2x', aggregate['selected_frame_totals']['v2x'])
    )
    delta_stats = aggregate['selected_frame_totals']['delta_v2x_minus_ego']
    print(
        '  '
        f"delta_v2x_minus_ego: success_gt={delta_stats['success_gt']}, "
        f"success_dt={delta_stats['success_dt']}, "
        f"object_missing={delta_stats['object_missing']}, "
        f"false_detection={delta_stats['false_detection']}, "
        f"localization_failure={delta_stats['localization_failure']} "
        f"(gt={delta_stats['localization_failure_gt']}, "
        f"dt={delta_stats['localization_failure_dt']})"
    )
    print(f"  matched_transitions={aggregate['matched_transition_counts']}")

    if missing:
        print()
        print('[missing]')
        for path_text in missing:
            print(f'  {path_text}')

    if args.output_json is not None:
        payload = {
            'selection': {
                'range_name': args.range_name,
                'iou_key': iou_key,
                'source_status': args.source_status,
                'target_statuses': target_statuses,
            },
            'models': results,
            'aggregate': aggregate,
            'missing': missing,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        print()
        print(f'JSON saved to {args.output_json}')


if __name__ == '__main__':
    main()
