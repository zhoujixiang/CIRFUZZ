

import argparse
import csv
import os
import sys
from collections import Counter, OrderedDict

RANGES = ['overall', 'short', 'middle', 'long']
STATUSES = ['success', 'object_missing', 'localization_failure']

TRANSITIONS = [
    'success -> success',
    'success -> object_missing',
    'success -> localization_failure',
    'object_missing -> success',
    'object_missing -> object_missing',
    'object_missing -> localization_failure',
    'localization_failure -> success',
    'localization_failure -> object_missing',
    'localization_failure -> localization_failure',
]

LOST = {'success -> object_missing', 'success -> localization_failure'}
RECOVERED = {'object_missing -> success', 'localization_failure -> success'}

DEFAULT_PAIRS = [
    ('attfuse_8_17', 'attfuse_ego_8_26'),
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
        'model_dirs',
        nargs='*',
        help='Model directories to analyze. Empty (default) uses --pairs.')
    parser.add_argument(
        '--pairs',
        action='store_true',
        help='Use the 9 built-in baseline/ego-preserving pairs (default).')
    parser.add_argument(
        '--model-root',
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model'),
        help='Root directory holding the model dirs.')
    parser.add_argument('--range-name', default='overall', choices=RANGES)
    parser.add_argument('--iou', default='0.70',
                        choices=['0.50', '0.70'],
                        help='IoU threshold, zero padded to 2 decimals.')
    parser.add_argument(
        '--no-common-frames', action='store_true',
        help=('Count every frame of each model instead of only the frames '
              'shared by all requested models.'))
    parser.add_argument('--csv', default=None, help='Write the matrix as CSV.')
    parser.add_argument('--json', default=None, help='Write raw counts as JSON.')
    return parser


def resolve_models(args):
    if args.model_dirs:
        return [(None, p) for p in args.model_dirs]
    root = os.path.abspath(args.model_root)
    out = []
    for base, ego in DEFAULT_PAIRS:
        out.append((base, os.path.join(root, base)))
        out.append((ego, os.path.join(root, ego)))
    return out


def count_transitions(path, range_name, iou_key, frame_filter=None):
    counter = Counter()
    frame_count = 0
    frames_with_difference = 0
    in_box = False
    in_range = False
    in_iou = False
    cur_ego = None
    cur_idx = -1
    keep_frame = frame_filter is None

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.rstrip('\n')
            indent = len(text) - len(text.lstrip(' '))
            body = text[indent:]
            if not body:
                continue

            if indent == 0:
                in_box = in_range = in_iou = False
                cur_ego = None
                if body.startswith('- '):
                    frame_count += 1
                    cur_idx = int(body[2:].split(':', 1)[1])
                    keep_frame = frame_filter is None or cur_idx in frame_filter
                continue

            if indent == 2:
                in_box = body.startswith('box_level_differences:')
                in_range = in_iou = False
                continue

            if not in_box:
                continue

            if indent == 4:
                in_range = (body == range_name + ':')
                in_iou = False
                continue

            if not in_range:
                continue

            if indent == 6:
                in_iou = keep_frame and (body == iou_key + ':')
                if in_iou:
                    frames_with_difference += 1
                continue

            if not in_iou or indent != 10:
                continue

            key, _, value = body.partition(':')
            key = key.strip()
            value = value.strip()
            if key == 'ego_status':
                cur_ego = value
            elif key == 'v2x_status':
                if cur_ego is not None:
                    counter['%s -> %s' % (cur_ego, value)] += 1
                cur_ego = None

    return counter, frame_count, frames_with_difference


def list_frame_indices(path):
    out = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('- frame_index:'):
                out.add(int(line.split(':', 1)[1]))
    return out


def main():
    args = build_parser().parse_args()
    iou_key = 'iou_%.2f' % float(args.iou)
    models = resolve_models(args)

    paths = []
    for name, model_dir in models:
        label = name or os.path.basename(os.path.normpath(model_dir))
        path = os.path.join(model_dir, 'compare_ego_v2x',
                            'different_cases.yaml')
        if not os.path.exists(path):
            print('[skip] %-24s no %s' % (label, path), file=sys.stderr)
            continue
        paths.append((label, path))

    common = None
    if not args.no_common_frames and len(paths) > 1:
        for _, path in paths:
            frames = list_frame_indices(path)
            common = frames if common is None else (common & frames)
        print('[common] %d frames shared by all %d models'
              % (len(common), len(paths)), file=sys.stderr)
    if args.no_common_frames:
        common = None

    results = OrderedDict()
    for label, path in paths:
        counter, frames, diff_frames = count_transitions(
            path, args.range_name, iou_key, frame_filter=common)
        total = sum(counter.values())
        lost = sum(counter[k] for k in LOST)
        recovered = sum(counter[k] for k in RECOVERED)
        results[label] = {
            'counter': counter,
            'total': total,
            'lost': lost,
            'recovered': recovered,
            'net': lost - recovered,
            'different_frames': diff_frames,
            'transition_frames': frames,
        }
        print('[read] %-24s objects=%d frames=%d' % (label, total, frames),
              file=sys.stderr)

    if not results:
        print('no different_cases.yaml found for the requested models')
        return 1

    cols = TRANSITIONS
    header = ['model'] + cols + ['total', 'lost', 'recovered', 'net']
    rows = []
    for label, r in results.items():
        c = r['counter']
        rows.append([label] + [c.get(k, 0) for k in cols] +
                    [r['total'], r['lost'], r['recovered'], r['net']])

    print('\n# ego -> v2x GT transitions (%s / %s)\n'
          % (args.range_name, iou_key))
    widths = [max(len(str(row[i])) for row in [header] + rows)
              for i in range(len(header))]
    fmt = ' | '.join('%%-%ds' % w for w in widths)
    print(fmt % tuple(header))
    print('-|-'.join('-' * w for w in widths))
    for row in rows:
        print(fmt % tuple(row))

    pairs = [(b, e) for b, e in DEFAULT_PAIRS
             if b in results and e in results]
    if pairs:
        print('\n## ego-preserving minus baseline (net = lost - recovered)\n')
        ph = ['pair', 'lost', 'recovered', 'net']
        prows = []
        for b, e in pairs:
            prows.append([
                '%s -> %s' % (b, e),
                results[e]['lost'] - results[b]['lost'],
                results[e]['recovered'] - results[b]['recovered'],
                results[e]['net'] - results[b]['net'],
            ])
        pw = [max(len(str(r[i])) for r in [ph] + prows)
              for i in range(len(ph))]
        pfmt = ' | '.join('%%-%ds' % w for w in pw)
        print(pfmt % tuple(ph))
        print('-|-'.join('-' * w for w in pw))
        for row in prows:
            print(pfmt % tuple(row))

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print('\nwrote %s' % args.csv)

    if args.json:
        import json
        payload = {
            'range': args.range_name,
            'iou': iou_key,
            'models': {k: {'counts': dict(v['counter']),
                           'total': v['total'],
                           'lost': v['lost'],
                           'recovered': v['recovered'],
                           'net': v['net']}
                       for k, v in results.items()},
        }
        with open(args.json, 'w') as f:
            json.dump(payload, f, indent=2)
        print('wrote %s' % args.json)

    return 0


if __name__ == '__main__':
    sys.exit(main())
