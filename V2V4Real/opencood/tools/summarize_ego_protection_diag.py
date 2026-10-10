import sys

import yaml


def _tuple_ctor(loader, node):
    return tuple(loader.construct_sequence(node))


yaml.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_ctor)


def _ap_scalar(v):
    if isinstance(v, (tuple, list)):
        return v[0]
    return v


def summarize(path):
    with open(path) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    print('=' * 78)
    print('model_dir: %s  (frames processed: %s)' % (
        d.get('model_dir', path), d.get('num_frames_processed')))
    print('=' * 78)

    print('\n-- Box displacement (ego-range association dist = %s) --' %
          d.get('ego_range_assoc_dist'))
    merged_in = d['merged_in_ego_range']
    ego_driven = d['ego_driven_in_ego_range']
    non_ego = d['non_ego_in_ego_range']
    for rng in merged_in:
        pct = 100.0 * ego_driven[rng] / merged_in[rng] \
            if merged_in[rng] else 0.0
        print('  %-8s merged-boxes-in-range=%-6d ego-driven=%-6d (%.1f%%) '
              'non-ego-boxes-in-range=%d' % (
                  rng, merged_in[rng], ego_driven[rng], pct, non_ego[rng]))

    print('\n-- AP: merged vs Plan-1 protected --')
    hdr = '  %-8s | %10s %10s %8s | %10s %10s %8s' % (
        'range', 'AP50-mrg', 'AP50-prot', 'd50', 'AP70-mrg', 'AP70-prot',
        'd70')
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for rng in ['overall', 'short', 'middle', 'long']:
        am = d['ap_merged'][rng]
        ap = d['ap_protected_plan1'][rng]
        a50m = _ap_scalar(am['0.5']); a50p = _ap_scalar(ap['0.5'])
        a70m = _ap_scalar(am['0.7']); a70p = _ap_scalar(ap['0.7'])
        print('  %-8s | %10.4f %10.4f %+8.4f | %10.4f %10.4f %+8.4f' % (
            rng, a50m, a50p, a50p - a50m, a70m, a70p, a70p - a70m))

    if 'error_merged' in d and 'error_protected_plan1' in d:
        print('\n-- Error-type counts (ego-lidar visibility filtered) --')
        em = d['error_merged']
        ep = d['error_protected_plan1']
        for rng in ['overall', 'short', 'middle', 'long']:
            for iou in ['0.5', '0.7']:
                key = 'iou_%.2f' % float(iou)
                m = em[rng][key]
                p = ep[rng][key]
                print('  %-8s iou%s: success_gt %5d->%5d (%+d) | '
                      'obj_missing %5d->%5d (%+d) | '
                      'loc_fail_gt %5d->%5d (%+d) | '
                      'false_det %5d->%5d (%+d)' % (
                          rng, iou,
                          m['success_gt'], p['success_gt'],
                          p['success_gt'] - m['success_gt'],
                          m['object_missing'], p['object_missing'],
                          p['object_missing'] - m['object_missing'],
                          m['localization_failure_gt'],
                          p['localization_failure_gt'],
                          p['localization_failure_gt'] -
                          m['localization_failure_gt'],
                          m['false_detection'], p['false_detection'],
                          p['false_detection'] - m['false_detection']))
    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        summarize(p)
