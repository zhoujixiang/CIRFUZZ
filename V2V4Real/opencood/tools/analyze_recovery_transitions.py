
import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_MODEL_NAMES = [
    'attfuse_8_17',
    'coalign_8_09',
    'CoBEVT',
    'cost',
    'early_fusion',
    'Fcooper',
    'pointpillar_late',
    'second_late',
    'V2XViT',
    'second_early',
]

RANGE_CHOICES = ['overall', 'short', 'middle', 'long']
IOU_CHOICES = ['0.50', '0.70']


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'model_dirs',
        nargs='*',
        help=(
            'Model directories to analyze. If omitted, use the default model '
            'directories under --model-root.'
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
        help='Ego status used to select degradation frames.',
    )
    parser.add_argument(
        '--target-statuses',
        default='object_missing,localization_failure',
        help=(
            'Comma separated v2x statuses used to select degradation frames. '
            'Default matches the previous analysis.'
        ),
    )
    parser.add_argument(
        '--all-frames',
        action='store_true',
        help=(
            'Count recoveries on every frame in different_cases.yaml instead '
            'of only selected degradation frames.'
        ),
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        default=None,
        help='Optional path to save the full summary as JSON.',
    )
    return parser


def normalize_statuses(status_text):
    statuses = [item.strip() for item in status_text.split(',') if item.strip()]
    if not statuses:
        raise ValueError('At least one target status must be provided.')
    return statuses


def resolve_model_dirs(args):
    if args.model_dirs:
        return [Path(p).resolve() for p in args.model_dirs]
    return [(args.model_root / name).resolve() for name in DEFAULT_MODEL_NAMES]


def finalize_frame(transitions, result, source_status, target_statuses,
                   count_all_frames):
    if not transitions:
        return

    selected = count_all_frames or any(
        ego == source_status and v2x in target_statuses
        for ego, v2x in transitions
    )
    if not selected:
        return

    result['counted_frame_count'] += 1
    frame_counter = Counter('%s -> %s' % item for item in transitions)
    result['transition_counts'].update(frame_counter)

    for ego_status, v2x_status in transitions:
        if ego_status != 'success':
            result['ego_error_total'] += 1
            if v2x_status == 'success':
                result['ego_error_to_v2x_success'] += 1
                result['recovery_counts']['%s -> success' % ego_status] += 1
        if ego_status == 'success' and v2x_status != 'success':
            result['ego_success_to_v2x_error'] += 1
            result['degradation_counts']['success -> %s' % v2x_status] += 1

        if ego_status == source_status and v2x_status in target_statuses:
            result['selected_object_count'] += 1
            result['matched_transition_counts'][
                '%s -> %s' % (ego_status, v2x_status)
            ] += 1


def scan_transitions(different_cases_path, range_name, iou_key, source_status,
                     target_statuses, count_all_frames):
    result = {
        'case_count': 0,
        'counted_frame_count': 0,
        'selected_object_count': 0,
        'ego_error_total': 0,
        'ego_error_to_v2x_success': 0,
        'ego_success_to_v2x_error': 0,
        'transition_counts': Counter(),
        'matched_transition_counts': Counter(),
        'recovery_counts': Counter(),
        'degradation_counts': Counter(),
    }

    in_box_level = False
    current_range = None
    current_iou = None
    in_gt_transitions = False
    pending_ego_status = None
    frame_transitions = []

    with different_cases_path.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')

            if line.startswith('- frame_index:'):
                finalize_frame(
                    frame_transitions,
                    result,
                    source_status,
                    target_statuses,
                    count_all_frames,
                )
                result['case_count'] += 1
                frame_transitions = []
                in_box_level = False
                current_range = None
                current_iou = None
                in_gt_transitions = False
                pending_ego_status = None
                continue

            if line == '  box_level_differences:':
                in_box_level = True
                current_range = None
                current_iou = None
                in_gt_transitions = False
                pending_ego_status = None
                continue

            if not in_box_level:
                continue

            if line.startswith('    ') and not line.startswith('      '):
                current_range = line.strip().rstrip(':')
                current_iou = None
                in_gt_transitions = False
                pending_ego_status = None
                continue

            if current_range == range_name and line.startswith('      ') \
                    and not line.startswith('        '):
                current_iou = line.strip().rstrip(':')
                in_gt_transitions = False
                pending_ego_status = None
                continue

            if current_range != range_name or current_iou != iou_key:
                continue

            if line.startswith('        gt_transitions:'):
                in_gt_transitions = True
                pending_ego_status = None
                continue

            if in_gt_transitions and line.startswith('        transition_counter:'):
                in_gt_transitions = False
                pending_ego_status = None
                continue

            if not in_gt_transitions:
                continue

            stripped = line.strip()
            if stripped.startswith('ego_status: '):
                pending_ego_status = stripped.split(': ', 1)[1]
            elif stripped.startswith('v2x_status: ') and pending_ego_status is not None:
                frame_transitions.append(
                    (pending_ego_status, stripped.split(': ', 1)[1])
                )
                pending_ego_status = None

    finalize_frame(
        frame_transitions,
        result,
        source_status,
        target_statuses,
        count_all_frames,
    )
    return result


def summarize_model(model_dir, args, iou_key, target_statuses):
    different_cases_path = model_dir / 'compare_ego_v2x' / 'different_cases.yaml'
    result = scan_transitions(
        different_cases_path=different_cases_path,
        range_name=args.range_name,
        iou_key=iou_key,
        source_status=args.source_status,
        target_statuses=set(target_statuses),
        count_all_frames=args.all_frames,
    )

    recovered = result['ego_error_to_v2x_success']
    ego_error_total = result['ego_error_total']
    degraded = result['ego_success_to_v2x_error']
    result.update({
        'model_name': model_dir.name,
        'model_dir': str(model_dir),
        'different_cases_path': str(different_cases_path),
        'recovery_rate': recovered / ego_error_total if ego_error_total else 0.0,
        'net_success_change_from_transitions': recovered - degraded,
    })
    return result


def counter_to_dict(counter):
    return dict(sorted(counter.items()))


def printable_result(result):
    output = dict(result)
    for key in [
            'transition_counts',
            'matched_transition_counts',
            'recovery_counts',
            'degradation_counts',
    ]:
        output[key] = counter_to_dict(output[key])
    return output


def aggregate_results(results):
    aggregate = {
        'model_name': 'aggregate',
        'case_count': 0,
        'counted_frame_count': 0,
        'selected_object_count': 0,
        'ego_error_total': 0,
        'ego_error_to_v2x_success': 0,
        'ego_success_to_v2x_error': 0,
        'transition_counts': Counter(),
        'matched_transition_counts': Counter(),
        'recovery_counts': Counter(),
        'degradation_counts': Counter(),
    }

    for result in results:
        for key in [
                'case_count',
                'counted_frame_count',
                'selected_object_count',
                'ego_error_total',
                'ego_error_to_v2x_success',
                'ego_success_to_v2x_error',
        ]:
            aggregate[key] += result[key]
        for key in [
                'transition_counts',
                'matched_transition_counts',
                'recovery_counts',
                'degradation_counts',
        ]:
            aggregate[key].update(result[key])

    recovered = aggregate['ego_error_to_v2x_success']
    ego_error_total = aggregate['ego_error_total']
    degraded = aggregate['ego_success_to_v2x_error']
    aggregate['recovery_rate'] = (
        recovered / ego_error_total if ego_error_total else 0.0
    )
    aggregate['net_success_change_from_transitions'] = recovered - degraded
    return aggregate


def print_result(result):
    print(f"[{result['model_name']}]")
    print(
        '  '
        f"counted_frames={result['counted_frame_count']}, "
        f"selected_objects={result['selected_object_count']}, "
        f"different_cases={result['case_count']}"
    )
    print(
        '  '
        f"ego_error_total={result['ego_error_total']}, "
        f"ego_error_to_v2x_success={result['ego_error_to_v2x_success']}, "
        f"recovery_rate={result['recovery_rate']:.4%}"
    )
    print(
        '  '
        f"ego_success_to_v2x_error={result['ego_success_to_v2x_error']}, "
        f"net_success_change={result['net_success_change_from_transitions']}"
    )
    print(
        '  '
        f"recovery_counts={counter_to_dict(result['recovery_counts'])}"
    )
    print(
        '  '
        f"matched_degradation_counts="
        f"{counter_to_dict(result['matched_transition_counts'])}"
    )
    print()


def main():
    args = build_parser().parse_args()
    iou_key = f'iou_{args.iou}'
    target_statuses = normalize_statuses(args.target_statuses)
    model_dirs = resolve_model_dirs(args)

    print(
        'Selection: '
        f"range={args.range_name}, iou={iou_key}, "
        f"source_status={args.source_status}, "
        f"target_statuses={target_statuses}, "
        f"scope={'all_frames' if args.all_frames else 'selected_frames'}"
    )
    print()

    results = []
    missing = []
    for model_dir in model_dirs:
        different_cases_path = model_dir / 'compare_ego_v2x' / 'different_cases.yaml'
        if not different_cases_path.exists():
            missing.append(str(different_cases_path))
            continue
        results.append(summarize_model(model_dir, args, iou_key, target_statuses))

    for result in results:
        print_result(result)

    aggregate = aggregate_results(results)
    print_result(aggregate)

    if missing:
        print('[missing]')
        for path_text in missing:
            print(f'  {path_text}')
        print()

    if args.output_json is not None:
        payload = {
            'selection': {
                'range_name': args.range_name,
                'iou_key': iou_key,
                'source_status': args.source_status,
                'target_statuses': target_statuses,
                'scope': 'all_frames' if args.all_frames else 'selected_frames',
            },
            'models': [printable_result(result) for result in results],
            'aggregate': printable_result(aggregate),
            'missing': missing,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        print(f'JSON saved to {args.output_json}')


if __name__ == '__main__':
    main()
