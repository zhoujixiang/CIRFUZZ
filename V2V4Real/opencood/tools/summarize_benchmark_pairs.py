#!/usr/bin/env python

import argparse
import json
from pathlib import Path

PAIRS = [
    ('CoAlign',          'coalign_8_09',        'coalign_ego_8_09'),
    ('CoBEVT',           'CoBEVT',              'CoBEVT_ego_8_30'),
    ('cost',             'cost',                'cost_ego'),
    ('early_fusion',     'early_fusion',        'early_fusion_ego'),
    ('Fcooper',          'Fcooper',             'Fcooper_ego'),
    ('pointpillar_late', 'pointpillar_late',    'pointpillar_late_ego'),
    ('second_early',     'second_early',        'second_early_ego'),
    ('second_late',      'second_late',         'second_late_ego'),
]


def load(bench_dir, name):
    p = bench_dir / f'{name}.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt(v):
    return f'{v:8.2f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench_dir', type=Path,
                    default=Path(__file__).resolve().parents[1] / 'bench_inference_time')
    ap.add_argument('--output_md', type=Path, default=None)
    args = ap.parse_args()

    lines = []
    hdr = (f"{'pair':<18}{'model':<24}{'fwd(ms)':>9}{'post(ms)':>9}"
           f"{'e2e(ms)':>9}{'FPS':>8}{'ckpt':>6}")
    lines.append(hdr)
    lines.append('-' * len(hdr))

    for disp, orig, ego in PAIRS:
        for role, name in (('orig', orig), ('ego', ego)):
            d = load(args.bench_dir, name)
            if d is None:
                lines.append(f'{disp:<18}{name:<24}{"MISSING":>9}')
                continue
            r = d['result']
            e2e = r['e2e_ms']['mean']
            lines.append(
                f'{disp if role=="orig" else "":<18}'
                f'{name:<24}'
                f'{fmt(r["forward_ms"]["mean"])}'
                f'{fmt(r["post_ms"]["mean"])}'
                f'{fmt(e2e)}'
                f'{1000.0/e2e:8.1f}'
                f'{d["checkpoint_epoch"]:6d}'
            )
        do = load(args.bench_dir, orig)
        de = load(args.bench_dir, ego)
        if do and de:
            eo = do['result']['e2e_ms']['mean']
            ee = de['result']['e2e_ms']['mean']
            fo = do['result']['forward_ms']['mean']
            fe = de['result']['forward_ms']['mean']
            lines.append(
                f'{"  Δ ego-orig":<18}{"":<24}{fe-fo:+9.2f}{"":>9}'
                f'{ee-eo:+9.2f}{(1000.0/ee)-(1000.0/eo):+8.1f}'
            )
        lines.append('')

    text = '\n'.join(lines)
    print(text)
    if args.output_md:
        args.output_md.write_text('```\n' + text + '\n```\n')
        print(f'\nwrote {args.output_md}')


if __name__ == '__main__':
    main()
