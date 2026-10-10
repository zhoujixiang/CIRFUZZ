
import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


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

FRAME_ERROR_KEYS = ['object_missing', 'false_detection', 'localization_failure']

RANGE_CHOICES = ['overall', 'short', 'middle', 'long']
IOU_CHOICES = ['0.50', '0.70']

MODE_DIR_NAMES = ['fuzz', 'random']

EVAL_DIR_NAMES = ['ego_only', 'v2x', 'v2x_fullscene']

EVAL_FILENAMES = ['eval.yaml']
RANGE_EVAL_FILENAMES = [f'{r}_eval.yaml' for r in RANGE_CHOICES if r != 'overall']


def get_yaml_loader():
    return getattr(yaml, 'CSafeLoader', yaml.SafeLoader)


def load_yaml(path):
    if not Path(path).exists():
        return None
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.load(f, Loader=get_yaml_loader())


def load_yaml_list(yaml_path):
    data = load_yaml(yaml_path)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f'{yaml_path} is expected to contain a YAML list.')
    return data


def _find_analysis_root(model_dir):
    nested = model_dir / 'compare_ego_v2x'
    if nested.exists() and nested.is_dir():
        return nested, 'nested'
    return model_dir, 'flat'


def _find_mode_root(mode_dir):
    nested = mode_dir / 'compare_ego_v2x'
    if nested.exists() and nested.is_dir():
        return nested, 'nested'
    return mode_dir, 'flat'


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            '统计 fuzzing/compare_eval/results 下各模型在 fuzz / random '
            '两种评测下的检测质量与错误分析结果。'
        )
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        required=True,
        help=(
            'compare_eval/results 目录, 例如 '
            'Data/generated_data/early_fuzzing_coop/compare_eval/results'
        ),
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=None,
        help=(
            '要分析的模型子目录名; 缺省时扫描 --results-dir 下所有包含 '
            '评测结果的子目录。'
        ),
    )
    parser.add_argument(
        '--modes',
        nargs='*',
        default=MODE_DIR_NAMES,
        choices=MODE_DIR_NAMES,
        help='要统计的运行模式目录名 (默认 fuzz random)。',
    )
    parser.add_argument(
        '--range-name',
        choices=RANGE_CHOICES,
        default='overall',
        help='距离范围, 用于错误统计与逐范围 AP。',
    )
    parser.add_argument(
        '--iou',
        choices=IOU_CHOICES,
        default='0.70',
        help='IoU 阈值, 用于错误统计。',
    )
    parser.add_argument(
        '--source-status',
        default='success',
        help='box_level 选择目标 GT 对象时要求的 ego 状态。',
    )
    parser.add_argument(
        '--target-statuses',
        default='object_missing',
        help=(
            '逗号分隔的 v2x 状态列表, 满足 (ego==source_status, '
            'v2x==target_status) 的 GT 对象被计为失败, 例如 object_missing '
            '或 object_missing,localization_failure。'
        ),
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        default=None,
        help='可选: 将完整汇总结果保存为 JSON。',
    )
    parser.add_argument(
        '--groups-path',
        type=Path,
        default=None,
        help=(
            '分组文件的路径; 文件内容为每一行一个目录名, 支持通配符 '
            '(例如 /path/to/early_fuzz_*.txt)。用于在多个 results 目录间 '
            '汇总同一个模型。'
        ),
    )
    parser.add_argument(
        '--case-file',
        type=str,
        default='different_cases.yaml',
        help='逐帧案例文件名 (用于错误统计与转换计数), 默认 different_cases.yaml。',
    )
    parser.add_argument(
        '--show-frame-indices',
        action='store_true',
        help='对每个模型打印被选中的帧索引。',
    )
    parser.add_argument(
        '--no-horizontal',
        action='store_true',
        help='不打印模型间的横向对比表 (默认打印)。',
    )
    return parser


def make_stats_template():
    return {key: 0 for key in ERROR_KEYS}


def add_error_stats(accumulator, stats):
    if not stats:
        return
    for key in ERROR_KEYS:
        accumulator[key] += int(stats.get(key, 0))


def finalize_error_stats(stats):
    output = dict(stats)
    output['localization_failure'] = (
        int(output.get('localization_failure_gt', 0)) +
        int(output.get('localization_failure_dt', 0))
    )
    return output


def normalize_statuses(status_text):
    statuses = [item.strip() for item in status_text.split(',') if item.strip()]
    if not statuses:
        raise ValueError('At least one target status must be provided.')
    return statuses


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


def analyze_cases(cases, range_name, iou_key, source_status, target_statuses):
    selected_frames = []
    transition_counter = Counter()
    total_transition_counter = Counter()
    mode_totals = {
        'ego_only': make_stats_template(),
        'v2x': make_stats_template(),
        'delta_v2x_minus_ego': {key: 0 for key in ERROR_KEYS
                                if key not in {'gt_total', 'dt_total'}},
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
            'delta_v2x_minus_ego': {key: 0 for key in ERROR_KEYS
                                    if key not in {'gt_total', 'dt_total'}},
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
        for key, value in result['selected_frame_totals'][
                'delta_v2x_minus_ego'].items():
            if key in aggregate['selected_frame_totals'][
                    'delta_v2x_minus_ego']:
                aggregate['selected_frame_totals'][
                    'delta_v2x_minus_ego'][key] += int(value)
        aggregate['matched_transition_counts'].update(
            result['matched_transition_counts']
        )

    aggregate['selected_frame_totals']['ego_only'] = finalize_error_stats(
        aggregate['selected_frame_totals']['ego_only']
    )
    aggregate['selected_frame_totals']['v2x'] = finalize_error_stats(
        aggregate['selected_frame_totals']['v2x']
    )
    aggregate['selected_frame_totals']['delta_v2x_minus_ego'] = (
        finalize_error_stats(
            aggregate['selected_frame_totals']['delta_v2x_minus_ego']
        )
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


def collect_ap_metrics(mode_dir, filenames):
    metrics = {}
    for eval_dir_name in EVAL_DIR_NAMES:
        eval_dir = mode_dir / eval_dir_name
        if not eval_dir.is_dir():
            continue
        for filename in filenames:
            data = load_yaml(eval_dir / filename)
            if data is None:
                continue
            label = eval_dir_name if filename == 'eval.yaml' else (
                f'{eval_dir_name}/{Path(filename).stem}'
            )
            metrics[label] = {
                key: float(data[key]) for key in ('ap_50', 'ap_70')
                if key in data
            }
    return metrics


def collect_error_summary(mode_dir, range_name, iou_key):
    data = load_yaml(mode_dir / 'error_summary_compare.yaml')
    if data is None:
        return None
    summary = {}
    for mode in ('ego_only', 'v2x', 'delta_v2x_minus_ego'):
        stats = (data.get(mode, {}).get(range_name, {}) or {}).get(iou_key)
        if stats is None:
            continue
        filtered = {key: stats[key] for key in ERROR_KEYS if key in stats}
        summary[mode] = finalize_error_stats(filtered)
    return summary or None


def collect_compare_diff(mode_dir):
    data = load_yaml(mode_dir / 'compare_error_diff.yaml')
    if data is None:
        return None
    return {
        key: data[key] for key in
        ('frame_count', 'different_frame_count', 'difference_frame_rate')
        if key in data
    }


def analyze_mode(mode_dir, args, filenames):
    mode_name = mode_dir.name
    root_dir, layout = _find_mode_root(mode_dir)
    case_file = root_dir / args.case_file
    if not case_file.exists():
        return None

    cases = load_yaml_list(case_file)
    case_analysis = analyze_cases(
        cases,
        args.range_name,
        f'iou_{args.iou}',
        args.source_status,
        args.target_statuses,
    )

    return {
        'mode': mode_name,
        'case_file': str(case_file),
        'frame_count': len(cases),
        'layout': layout,
        'ap_metrics': collect_ap_metrics(root_dir, filenames),
        'error_summary': collect_error_summary(root_dir, args.range_name,
                                               f'iou_{args.iou}'),
        'compare_diff': collect_compare_diff(root_dir),
        **case_analysis,
    }


def analyze_model(model_dir, args, filenames):
    root_dir, layout = _find_analysis_root(model_dir)
    mode_results = []
    for mode_name in args.modes:
        mode_dir = root_dir / mode_name
        if not mode_dir.is_dir():
            continue
        result = analyze_mode(mode_dir, args, filenames)
        if result is not None:
            result['model_layout'] = layout
            mode_results.append(result)

    if not mode_results:
        return None

    model_name = model_dir.name
    return {
        'model_name': model_name,
        'model_dir': str(model_dir),
        'layout': layout,
        'modes': mode_results,
    }


def resolve_group_paths(groups_path):
    import glob as globlib

    groups_path = Path(groups_path)
    if not groups_path.exists():
        raise FileNotFoundError(f'groups file not found: {groups_path}')
    model_dirs = []
    for line in groups_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('/'):
            matches = globlib.glob(line)
        else:
            matches = globlib.glob(str(Path('.') / line))
        for match in sorted(matches):
            model_dirs.append(Path(match).resolve())
    return model_dirs


def resolve_model_dirs(results_dir, model_names, groups_path):
    if groups_path is not None:
        return resolve_group_paths(groups_path)
    if model_names:
        return [(results_dir / name).resolve() for name in model_names]
    return sorted(
        child for child in results_dir.iterdir()
        if child.is_dir() and _has_analysis_data(child)
    )


def _has_analysis_data(model_dir):
    root_dir, _ = _find_analysis_root(model_dir)
    return any((root_dir / name).is_dir() for name in MODE_DIR_NAMES)


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


def format_ap_row(model, mode):
    metrics = mode.get('ap_metrics', {})
    if not metrics:
        return []
    rows = []
    for label in sorted(metrics):
        ap = metrics[label]
        ap_50 = ap.get('ap_50', float('nan'))
        ap_70 = ap.get('ap_70', float('nan'))
        rows.append(
            f"  {label}: ap_50={ap_50:.4f}, ap_70={ap_70:.4f}"
        )
    return rows


EGO_V2X_METRIC_KEYS = [
    'ap_50',
    'ap_70',
    'success_gt',
    'success_dt',
    'object_missing',
    'false_detection',
    'localization_failure',
]

EGO_V2X_METRICS = ['ap_50', 'ap_70', 'success_gt', 'object_missing',
                   'false_detection', 'localization_failure']


def _as_num(value):
    if value is None:
        return float('nan')
    return float(value)


def get_ego_v2x_table(mode):
    table = {}
    ap_metrics = mode.get('ap_metrics', {})
    error_summary = mode.get('error_summary', {}) or {}

    for metric in EGO_V2X_METRICS:
        if metric in ('ap_50', 'ap_70'):
            ego = ap_metrics.get('ego_only', {}).get(metric)
            v2x = ap_metrics.get('v2x', {}).get(metric)
        else:
            ego_stats = error_summary.get('ego_only', {})
            v2x_stats = error_summary.get('v2x', {})
            ego = ego_stats.get(metric)
            v2x = v2x_stats.get(metric)
        ego_num = _as_num(ego)
        v2x_num = _as_num(v2x)
        table[metric] = {
            'ego': ego_num,
            'v2x': v2x_num,
            'delta': v2x_num - ego_num,
        }
    return table


def _fmt_delta(delta, is_rate=False):
    if is_rate:
        return f'{delta * 100:+.1f}pp'
    return f'{int(delta):+d}'


def print_ego_v2x_compare_table(mode):
    table = get_ego_v2x_table(mode)
    print(f"    {'metric':<20}{'ego':>12}{'v2x':>12}{'delta(v2x-ego)':>16}")
    for metric in EGO_V2X_METRICS:
        row = table[metric]
        if metric in ('ap_50', 'ap_70'):
            print(
                f"    {metric:<20}{row['ego']:>12.4f}{row['v2x']:>12.4f}"
                f"{_fmt_delta(row['delta'], is_rate=True):>16}"
            )
        else:
            print(
                f"    {metric:<20}{int(row['ego']):>12d}{int(row['v2x']):>12d}"
                f"{_fmt_delta(row['delta']):>16}"
            )


def get_metric_value(mode, metric, ap_metric):
    if metric in ('ap_50', 'ap_70'):
        return mode.get('ap_metrics', {}).get(ap_metric, {}).get(metric)
    error_summary = mode.get('error_summary', {}) or {}
    return error_summary.get(ap_metric, {}).get(metric)


def print_horizontal_compare(results, mode_name, ap_metrics, metrics):
    model_names = [model['model_name'] for model in results]

    for ap_metric in ap_metrics:
        metric_cells = {}
        for metric in metrics:
            is_rate = metric in ('ap_50', 'ap_70')
            cells = []
            for model in results:
                mode = next(
                    (m for m in model['modes'] if m['mode'] == mode_name),
                    None,
                )
                value = (
                    get_metric_value(mode, metric, ap_metric)
                    if mode is not None else None
                )
                cells.append(_fmt_value(value, is_rate))
            metric_cells[metric] = cells

        metric_width = max(6, *[len(m) for m in metrics])
        col_widths = [
            max(len(name),
                *[len(metric_cells[m][i]) for m in metrics])
            for i, name in enumerate(model_names)
        ]
        col_widths = [max(w, 10) for w in col_widths]

        print(f'    [{mode_name} | {ap_metric}]')
        header_cells = [f'{name:<{w}}' for name, w in zip(model_names, col_widths)]
        print('      ' + f'{"metric":<{metric_width + 2}}' + '  '.join(header_cells))
        for metric in metrics:
            cells = metric_cells[metric]
            aligned = [
                cell if cell == 'n/a' else cell.rjust(w)
                for cell, w in zip(cells, col_widths)
            ]
            row_cells = [
                f'{c:<{w}}' for c, w in zip(aligned, col_widths)
            ]
            print(
                '      '
                + f'{metric:<{metric_width + 2}}'
                + '  '.join(row_cells)
            )
        print()


def _fmt_value(value, is_rate=False):
    if value is None:
        return 'n/a'
    if is_rate:
        return f'{float(value):.4f}'
    return f'{int(value):d}'


def print_mode_result(model, mode, show_frame_indices=False):
    print(f"  [{mode['mode']}]")
    print(
        '    '
        f"frames={mode['frame_count']}, "
        f"selected_frames={mode['selected_frame_count']}, "
        f"selected_objects={mode['selected_object_count']}, "
        f"different_cases={mode['different_case_count']}"
    )
    print(
        '    '
        f"success_gt_change={mode['success_gt_change_in_selected_frames']}, "
        f"success_dt_change={mode['success_dt_change_in_selected_frames']}"
    )

    error_summary = mode.get('error_summary')
    if error_summary:
        print(
            '    ' +
            format_mode_stats('ego_only', error_summary['ego_only'])
        )
        print(
            '    ' +
            format_mode_stats('v2x', error_summary['v2x'])
        )
        delta_stats = error_summary.get('delta_v2x_minus_ego')
        if delta_stats:
            print(
                '    '
                f"delta_v2x_minus_ego: success_gt={delta_stats['success_gt']}, "
                f"success_dt={delta_stats['success_dt']}, "
                f"object_missing={delta_stats['object_missing']}, "
                f"false_detection={delta_stats['false_detection']}, "
                f"localization_failure={delta_stats['localization_failure']} "
                f"(gt={delta_stats['localization_failure_gt']}, "
                f"dt={delta_stats['localization_failure_dt']})"
            )

    compare_diff = mode.get('compare_diff')
    if compare_diff:
        print(
            '    '
            f"compare_diff: frame_count={compare_diff['frame_count']}, "
            f"different_frame_count={compare_diff['different_frame_count']}, "
            f"difference_frame_rate={compare_diff['difference_frame_rate']:.4f}"
        )

    print(f"    matched_transitions={mode['matched_transition_counts']}")
    for row in format_ap_row(model, mode):
        print(row)

    print('    ego vs v2x (内部对比):')
    print_ego_v2x_compare_table(mode)
    if show_frame_indices:
        print(f"    frame_indices={mode['selected_frame_indices']}")
    print()


def print_result(model, show_frame_indices=False):
    print(f"[{model['model_name']}]  (layout={model['layout']})")
    for mode in model['modes']:
        print_mode_result(model, mode, show_frame_indices=show_frame_indices)


def main():
    args = build_parser().parse_args()
    model_dirs = resolve_model_dirs(args.results_dir, args.models,
                                    args.groups_path)

    target_statuses = normalize_statuses(args.target_statuses)
    iou_key = f'iou_{args.iou}'

    if args.range_name == 'overall':
        filenames = list(EVAL_FILENAMES)
    else:
        filenames = [f'{args.range_name}_eval.yaml']

    print(
        'Selection: '
        f"results_dir={args.results_dir}, "
        f"range={args.range_name}, iou={iou_key}, "
        f"ego_status={args.source_status}, "
        f"v2x_statuses={target_statuses}"
    )
    print()

    results = []
    missing = []
    for model_dir in model_dirs:
        model_result = analyze_model(model_dir, args, filenames)
        if model_result is None:
            missing.append(str(model_dir))
            continue
        results.append(model_result)

    for model in results:
        print_result(model, show_frame_indices=args.show_frame_indices)

    if results and not args.no_horizontal:
        print('[horizontal compare]')
        for mode_name in args.modes:
            print_horizontal_compare(
                results,
                mode_name,
                ap_metrics=['ego_only', 'v2x'],
                metrics=['ap_50', 'ap_70', 'success_gt', 'object_missing',
                         'false_detection', 'localization_failure'],
            )

    if results:
        print('[aggregate]')
        for mode_name in args.modes:
            per_mode = [
                result for result in results
                if any(mode['mode'] == mode_name
                       for mode in result['modes'])
            ]
            if not per_mode:
                continue
            mode_results = [mode for result in per_mode
                            for mode in result['modes']
                            if mode['mode'] == mode_name]
            aggregate = aggregate_results(mode_results)
            print(f'  [{mode_name}]')
            print(
                '    '
                f"models={aggregate['model_count']}, "
                f"selected_frames={aggregate['selected_frame_count']}, "
                f"selected_objects={aggregate['selected_object_count']}, "
                f"different_cases={aggregate['different_case_count']}"
            )
            print(
                '    '
                f"success_gt_change={aggregate['success_gt_change_in_selected_frames']}, "
                f"success_dt_change={aggregate['success_dt_change_in_selected_frames']}"
            )
            print(
                '    ' +
                format_mode_stats(
                    'ego_only', aggregate['selected_frame_totals']['ego_only']
                )
            )
            print(
                '    ' +
                format_mode_stats(
                    'v2x', aggregate['selected_frame_totals']['v2x']
                )
            )
            delta_stats = aggregate['selected_frame_totals'][
                'delta_v2x_minus_ego'
            ]
            print(
                '    '
                f"delta_v2x_minus_ego: success_gt={delta_stats['success_gt']}, "
                f"success_dt={delta_stats['success_dt']}, "
                f"object_missing={delta_stats['object_missing']}, "
                f"false_detection={delta_stats['false_detection']}, "
                f"localization_failure={delta_stats['localization_failure']} "
                f"(gt={delta_stats['localization_failure_gt']}, "
                f"dt={delta_stats['localization_failure_dt']})"
            )
            print(
                f"    matched_transitions="
                f"{aggregate['matched_transition_counts']}"
            )
            print()

    if missing:
        print('[missing]')
        for path_text in missing:
            print(f'  {path_text}')

    if args.output_json is not None:
        payload = {
            'selection': {
                'results_dir': str(args.results_dir),
                'range_name': args.range_name,
                'iou_key': iou_key,
                'source_status': args.source_status,
                'target_statuses': target_statuses,
                'modes': args.modes,
                'groups_path': str(args.groups_path) if args.groups_path else None,
            },
            'models': results,
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
