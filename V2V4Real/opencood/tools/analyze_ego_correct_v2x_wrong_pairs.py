
import argparse
import json
from pathlib import Path

import yaml


IOU_CHOICES = ['0.50', '0.70']
RANGE_CHOICES = ['overall', 'short', 'middle', 'long']
SCOPE_CHOICES = ['aligned_frames', 'common_frames', 'all_frames']
FAILURE_STATUSES = ['object_missing', 'localization_failure']

MODEL_PAIRS = [
    ('coalign_8_09', 'coalign_ego_8_09'),
    ('CoBEVT', 'CoBEVT_ego_8_30'),
    ('cost', 'cost_ego'),
    ('early_fusion', 'early_fusion_ego'),
    ('Fcooper', 'Fcooper_ego'),
    ('pointpillar_late', 'pointpillar_late_ego'),
    ('second_early', 'second_early_ego'),
    ('second_late', 'second_late_ego'),
]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model-root',
        type=Path,
        default=Path(__file__).resolve().parents[2] / 'model',
        help='Root directory holding the model dirs.',
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
        help='Ego status that marks an object as "ego correct".',
    )
    parser.add_argument(
        '--target-statuses',
        default=','.join(FAILURE_STATUSES),
        help='Comma separated v2x statuses counted as "v2x wrong".',
    )
    parser.add_argument(
        '--scope',
        choices=SCOPE_CHOICES,
        default='aligned_frames',
        help='Frame scope used for the headline numbers.',
    )
    parser.add_argument(
        '--all-scopes',
        action='store_true',
        help='Also print the other two scopes for comparison.',
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Print the headline scope for every range and IoU combination.',
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        default=None,
        help='Optional path to save the full summary as JSON.',
    )
    parser.add_argument(
        '--emit-frames',
        type=Path,
        default=None,
        metavar='DIR',
        help=(
            'Write per-pair manifests of the underlying case dumps. For each '
            'pair a subdirectory is created with base.txt / ego.txt / '
            'repeat.txt / new.txt / repair.txt, each holding one case-dump '
            'prefix per line (feed the frame name to vis_ego_correct_v2x_wrong '
            'as --frame). Frames whose dump was never saved go to the matching '
            '*.nodump.txt instead of being silently dropped.'
        ),
    )
    return parser


def get_yaml_loader():
    return getattr(yaml, 'CSafeLoader', yaml.SafeLoader)


def load_yaml_list(path):
    with path.open('r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=get_yaml_loader())
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f'{path} is expected to contain a YAML list.')
    return data


def normalize_statuses(text):
    statuses = [item.strip() for item in text.split(',') if item.strip()]
    if not statuses:
        raise ValueError('At least one target status must be provided.')
    return statuses


def load_model(model_dir, range_name, iou_key, source_status):
    path = model_dir / 'compare_ego_v2x' / 'different_cases.yaml'
    if not path.exists():
        return None, path

    frames = {}
    for case in load_yaml_list(path):
        block = (
            case.get('box_level_differences', {})
            .get(range_name, {})
            .get(iou_key)
        )
        if block is None:
            continue
        transitions = block.get('gt_transitions') or []
        frames[int(case['frame_index'])] = {
            'gt_indices': {int(t['gt_index']) for t in transitions},
            'visible_gt_count': int(block.get('visible_gt_count', -1)),
            'source': {
                int(t['gt_index']): t['v2x_status']
                for t in transitions
                if t['ego_status'] == source_status
            },
        }
    return frames, path


def select_scope(base_frames, ego_frames, scope):
    if scope == 'all_frames':
        return base_frames, ego_frames

    common = set(base_frames) & set(ego_frames)
    if scope == 'common_frames':
        return (
            {f: base_frames[f] for f in common},
            {f: ego_frames[f] for f in common},
        )

    aligned = {
        f for f in common
        if base_frames[f]['gt_indices'] == ego_frames[f]['gt_indices']
        and base_frames[f]['visible_gt_count'] == ego_frames[f]['visible_gt_count']
    }
    return (
        {f: base_frames[f] for f in aligned},
        {f: ego_frames[f] for f in aligned},
    )


def collect_pairs(view, target_statuses):
    return {
        (frame, gt_index): status
        for frame, data in view.items()
        for gt_index, status in data['source'].items()
        if status in target_statuses
    }


def status_breakdown(pairs):
    breakdown = {status: 0 for status in FAILURE_STATUSES}
    for status in pairs.values():
        breakdown[status] = breakdown.get(status, 0) + 1
    return breakdown


def count_frames(pairs):
    return len({frame for frame, _ in pairs})


def analyze_pair(base_frames, ego_frames, target_statuses, scope):
    base_view, ego_view = select_scope(base_frames, ego_frames, scope)

    base_pairs = collect_pairs(base_view, target_statuses)
    ego_pairs = collect_pairs(ego_view, target_statuses)

    repeat_keys = set(ego_pairs) & set(base_pairs)
    new_keys = set(ego_pairs) - set(base_pairs)
    repair_keys = set(base_pairs) - set(ego_pairs)

    def frame_set(keys):
        return {frame for frame, _ in keys}

    return {
        'scope': scope,
        'base_frame_count': len(base_view),
        'ego_frame_count': len(ego_view),
        'base_object_count': len(base_pairs),
        'ego_object_count': len(ego_pairs),
        'repeat_object_count': len(repeat_keys),
        'new_object_count': len(new_keys),
        'repair_object_count': len(repair_keys),
        'base_status_breakdown': status_breakdown(base_pairs),
        'ego_status_breakdown': status_breakdown(ego_pairs),
        'repeat_status_breakdown': status_breakdown(
            {k: ego_pairs[k] for k in repeat_keys}
        ),
        'new_status_breakdown': status_breakdown(
            {k: ego_pairs[k] for k in new_keys}
        ),
        'repair_status_breakdown': status_breakdown(
            {k: base_pairs[k] for k in repair_keys}
        ),
        'base_frame_total': count_frames(base_pairs),
        'ego_frame_total': count_frames(ego_pairs),
        'repeat_frame_count': len(frame_set(repeat_keys)),
        'new_frame_count': len(frame_set(new_keys)),
        'repair_frame_count': len(frame_set(repair_keys)),
        'repeat_new_frame_overlap': len(
            frame_set(repeat_keys) & frame_set(new_keys)),
    }


def diagnose_alignment(base_frames, ego_frames):
    common = set(base_frames) & set(ego_frames)
    mismatched = [
        f for f in common
        if base_frames[f]['gt_indices'] != ego_frames[f]['gt_indices']
        or base_frames[f]['visible_gt_count'] != ego_frames[f]['visible_gt_count']
    ]
    return {
        'base_recorded_frames': len(base_frames),
        'ego_recorded_frames': len(ego_frames),
        'common_frames': len(common),
        'base_only_frames': len(set(base_frames) - set(ego_frames)),
        'ego_only_frames': len(set(ego_frames) - set(base_frames)),
        'gt_metadata_mismatch_frames': len(mismatched),
    }


def case_dir_of(model_dir):
    return model_dir / 'compare_ego_v2x' / 'cases'


def dump_available(model_dir, frame):
    return (case_dir_of(model_dir) / f'{frame:05d}_gt.npy').exists()


def partition_cases(base_frames, ego_frames, target_statuses, scope):
    base_view, ego_view = select_scope(base_frames, ego_frames, scope)
    base_pairs = collect_pairs(base_view, target_statuses)
    ego_pairs = collect_pairs(ego_view, target_statuses)

    overlap_keys = set(ego_pairs) & set(base_pairs)
    base_only_keys = set(base_pairs) - set(ego_pairs)
    new_keys = set(ego_pairs) - set(base_pairs)

    def frames(keys):
        return {frame for frame, _ in keys}

    overlap_frames = frames(overlap_keys)
    base_only_frames = frames(base_only_keys)
    return {
        'overlap': len(overlap_frames),
        'base_only': len(base_only_frames),
        'base_total': len(frames(base_pairs)),
        'new': len(frames(new_keys)),
        'ego_total': len(frames(ego_pairs)),
        'overlap_frames': overlap_frames,
        'base_only_frames': base_only_frames,
        'new_frames': frames(new_keys),
    }


def write_manifest(path, frames, model_dir):
    frames = sorted(frames)
    dumped = [f for f in frames if dump_available(model_dir, f)]
    nodump = [f for f in frames if f not in set(dumped)]

    case_dir = case_dir_of(model_dir)
    path.write_text(
        ''.join(f'{case_dir}/{f:05d}\n' for f in dumped), encoding='utf-8')
    if nodump:
        path.with_name(path.name + '.nodump.txt').write_text(
            ''.join(f'{f}\n' for f in nodump), encoding='utf-8')
    return len(dumped), len(nodump)


def emit_frame_manifests(out_root, pair_results, model_root, range_name,
                         iou_key, source_status, target_statuses, scope):
    out_root.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f'# scope={scope} range={range_name} iou={iou_key} '
                 f'ego={source_status} v2x_wrong={",".join(target_statuses)}')
    for entry in pair_results:
        base_dir = (model_root / entry['base_model']).resolve()
        ego_dir = (model_root / entry['ego_model']).resolve()
        pair_dir = out_root / f"{entry['base_model']}__{entry['ego_model']}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        cases = entry['case_partition'][scope]
        lines.append(f"[{entry['base_model']} -> {entry['ego_model']}]")
        for label, model_dir in [
                ('overlap', ego_dir),
                ('base_only', base_dir),
                ('new', ego_dir)]:
            frames = cases[f'{label}_frames']
            n_dump, n_nodump = write_manifest(
                pair_dir / f'{label}.txt', frames, model_dir)
            lines.append(f'  {label:9s} frames={len(frames):5d} '
                         f'dumps={n_dump:5d} missing_dump={n_nodump:4d}')
        lines.append('')
    (out_root / 'SUMMARY.txt').write_text('\n'.join(lines), encoding='utf-8')
    return lines


def run_analysis(model_root, range_name, iou_key, source_status, target_statuses,
                 scopes):
    per_pair = []
    missing = []
    for base_name, ego_name in MODEL_PAIRS:
        base_dir = (model_root / base_name).resolve()
        ego_dir = (model_root / ego_name).resolve()
        base_frames, base_path = load_model(
            base_dir, range_name, iou_key, source_status)
        if base_frames is None:
            missing.append(str(base_path))
            continue
        ego_frames, ego_path = load_model(
            ego_dir, range_name, iou_key, source_status)
        if ego_frames is None:
            missing.append(str(ego_path))
            continue
        per_pair.append({
            'base_dir': str(base_dir),
            'ego_dir': str(ego_dir),
            'base_model': base_name,
            'ego_model': ego_name,
            'alignment': diagnose_alignment(base_frames, ego_frames),
            'scopes': {
                scope: analyze_pair(
                    base_frames, ego_frames, target_statuses, scope)
                for scope in scopes
            },
            'case_partition': {
                scope: partition_cases(
                    base_frames, ego_frames, target_statuses, scope)
                for scope in scopes
            },
        })
    return per_pair, missing


OBJECT_KEYS = [
    'base_object_count',
    'ego_object_count',
    'repeat_object_count',
    'new_object_count',
    'repair_object_count',
]
FRAME_KEYS = [
    'base_frame_total',
    'ego_frame_total',
    'repeat_frame_count',
    'new_frame_count',
    'repair_frame_count',
    'repeat_new_frame_overlap',
]


def aggregate(pair_results, scope):
    total = {key: 0 for key in OBJECT_KEYS + FRAME_KEYS}
    total['repeat_status_breakdown'] = {s: 0 for s in FAILURE_STATUSES}
    total['new_status_breakdown'] = {s: 0 for s in FAILURE_STATUSES}
    total['repair_status_breakdown'] = {s: 0 for s in FAILURE_STATUSES}
    total['case_overlap'] = 0
    total['case_base_only'] = 0
    total['case_base_total'] = 0
    total['case_new'] = 0
    total['case_ego_total'] = 0
    for entry in pair_results:
        stats = entry['scopes'][scope]
        for key in OBJECT_KEYS + FRAME_KEYS:
            total[key] += stats[key]
        cases = entry['case_partition'][scope]
        for key in ['overlap', 'base_only', 'base_total', 'new', 'ego_total']:
            total[f'case_{key}'] += cases[key]
        for breakdown_key in [
                'repeat_status_breakdown',
                'new_status_breakdown',
                'repair_status_breakdown']:
            for status, value in stats[breakdown_key].items():
                total[breakdown_key][status] = (
                    total[breakdown_key].get(status, 0) + value
                )
    return total


def print_table(title, pair_results, scope):
    print(title)
    header = (
        f"{'model pair (base -> _ego)':38s} "
        f"{'OBJ ovl':>7s} {'OBJ bOnly':>9s} {'OBJ base':>8s} "
        f"{'OBJ new':>7s} {'OBJ ego':>7s} | "
        f"{'CAS ovl':>7s} {'CAS bOnly':>9s} {'CAS base':>8s} "
        f"{'CAS new':>7s} {'CAS ego':>7s}"
    )
    print(header)
    print('-' * len(header))

    def row(name, stats, cases):
        return (
            f'{name:38s} '
            f"{stats['repeat_object_count']:7d} "
            f"{stats['repair_object_count']:9d} "
            f"{stats['base_object_count']:8d} "
            f"{stats['new_object_count']:7d} "
            f"{stats['ego_object_count']:7d} | "
            f"{cases['overlap']:7d} "
            f"{cases['base_only']:9d} "
            f"{cases['base_total']:8d} "
            f"{cases['new']:7d} "
            f"{cases['ego_total']:7d}"
        )

    for entry in pair_results:
        name = f"{entry['base_model']} -> {entry['ego_model']}"
        if len(name) > 38:
            name = name[:35] + '...'
        print(row(name, entry['scopes'][scope], entry['case_partition'][scope]))
    total = aggregate(pair_results, scope)
    print('-' * len(header))
    print(row('AGGREGATE (sum over pairs)', total, {
        'overlap': total['case_overlap'],
        'base_only': total['case_base_only'],
        'base_total': total['case_base_total'],
        'new': total['case_new'],
        'ego_total': total['case_ego_total'],
    }))
    print(
        '  selection: ego_status=success, v2x_status in the failure set. '
        'ovl = failure in both runs; bOnly = only in base; new = only in _ego.'
    )
    print(
        '  overlap is one shared set, so both sides use the same ovl column:'
    )
    print('    base total = ovl + bOnly      ego total = ovl + new')
    print('    same identity holds for the CAS columns.')
    if total['base_object_count']:
        print(
            '  base composition: '
            f"ovl {total['repeat_object_count'] / total['base_object_count']:.2%}, "
            f"base-only {total['repair_object_count'] / total['base_object_count']:.2%}"
        )
    if total['ego_object_count']:
        print(
            '  _ego composition: '
            f"ovl {total['repeat_object_count'] / total['ego_object_count']:.2%}, "
            f"new {total['new_object_count'] / total['ego_object_count']:.2%}"
        )
    print('  ovl  by v2x status:', total['repeat_status_breakdown'])
    print('  bOnly by v2x status:', total['repair_status_breakdown'])
    print('  new  by v2x status:', total['new_status_breakdown'])
    print()


def print_alignment(pair_results):
    print('[GT metadata alignment per pair]')
    print(
        f"{'model pair':44s} {'baseF_rec':>9s} {'egoF_rec':>9s} "
        f"{'common':>7s} {'base_only':>10s} {'ego_only':>9s} {'gt_mismatch':>12s}"
    )
    for entry in pair_results:
        align = entry['alignment']
        name = f"{entry['base_model']} -> {entry['ego_model']}"
        if len(name) > 44:
            name = name[:41] + '...'
        flag = '  <-- gt_index NOT comparable' if align[
            'gt_metadata_mismatch_frames'] else ''
        print(
            f'{name:44s} '
            f"{align['base_recorded_frames']:9d} "
            f"{align['ego_recorded_frames']:9d} "
            f"{align['common_frames']:7d} "
            f"{align['base_only_frames']:10d} "
            f"{align['ego_only_frames']:9d} "
            f"{align['gt_metadata_mismatch_frames']:12d}{flag}"
        )
    print()


def main():
    args = build_parser().parse_args()
    iou_key = f'iou_{args.iou}'
    target_statuses = normalize_statuses(args.target_statuses)
    scopes = list(SCOPE_CHOICES)
    if not args.all_scopes:
        scopes = [args.scope]

    print(
        'Selection: '
        f'range={args.range_name}, iou={iou_key}, '
        f'ego_status={args.source_status}, v2x_statuses={target_statuses}'
    )
    print()

    if args.sweep:
        sweep = {}
        for range_name in RANGE_CHOICES:
            for iou in IOU_CHOICES:
                pair_results, _ = run_analysis(
                    args.model_root, range_name, f'iou_{iou}',
                    args.source_status, target_statuses, [args.scope])
                sweep[f'{range_name}/{iou}'] = pair_results
                print('=' * 100)
                print_table(
                    f"[range={range_name}, iou={iou}]", pair_results, args.scope)
        payload = {
            'selection': {
                'scope': args.scope,
                'source_status': args.source_status,
                'target_statuses': target_statuses,
            },
            'sweep': sweep,
        }
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding='utf-8',
            )
            print(f'JSON saved to {args.output_json}')
        return

    pair_results, missing = run_analysis(
        args.model_root, args.range_name, iou_key,
        args.source_status, target_statuses, scopes)

    print_alignment(pair_results)
    for scope in scopes:
        print_table(
            f'[scope={scope}]', pair_results, scope)

    if args.emit_frames is not None:
        lines = emit_frame_manifests(
            args.emit_frames, pair_results, args.model_root, args.range_name,
            iou_key, args.source_status, target_statuses, scopes[0])
        print(f'[case dumps written under {args.emit_frames}]')
        print('\n'.join(lines))
        print()

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
                'scopes': scopes,
            },
            'pairs': [
                {
                    **{k: v for k, v in entry.items() if k != 'case_partition'},
                    'case_partition': {
                        name: {k: v for k, v in cases.items()
                               if not k.endswith('_frames')}
                        for name, cases in entry['case_partition'].items()
                    },
                }
                for entry in pair_results
            ],
            'aggregate': {scope: aggregate(pair_results, scope) for scope in scopes},
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
