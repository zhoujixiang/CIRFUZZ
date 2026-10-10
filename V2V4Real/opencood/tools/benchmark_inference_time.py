#!/usr/bin/env python

import argparse
import json
import os
import statistics
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools import infrence_utils


def build_parser():
    parser = argparse.ArgumentParser(description='Benchmark per-frame inference time')
    parser.add_argument('--model_dir', required=True,
                        help='Path to the model log dir (holds config.yaml + *.pth)')
    parser.add_argument('--fusion_method', required=True,
                        choices=['nofusion', 'late', 'early', 'intermediate'])
    parser.add_argument('--max_frames', type=int, default=40,
                        help='Max number of frames to time (from the head of the test set)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Timed passes over the frame set; we report the mean of all passes')
    parser.add_argument('--warmup', type=int, default=20,
                        help='Warm-up e2e passes to run (and discard) before timing')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_json', default=None)
    return parser


def run_forward(batch_data, model, dataset, fusion_method):
    if fusion_method == 'nofusion':
        return infrence_utils.inference_no_fusion(batch_data, model, dataset)
    if fusion_method == 'late':
        return infrence_utils.inference_late_fusion(batch_data, model, dataset)
    if fusion_method == 'early':
        return infrence_utils.inference_early_fusion(batch_data, model, dataset)
    if fusion_method == 'intermediate':
        return infrence_utils.inference_intermediate_fusion(batch_data, model, dataset)
    raise NotImplementedError(fusion_method)


def forward_only(batch_data, model, fusion_method):
    output_dict = OrderedDict()
    if fusion_method == 'nofusion':
        for cav_id, cav_content in batch_data.items():
            if cav_id == 'ego':
                output_dict[cav_id] = model(cav_content)
    elif fusion_method == 'late':
        for cav_id, cav_content in batch_data.items():
            output_dict[cav_id] = model(cav_content)
    else:
        output_dict['ego'] = model(batch_data['ego'])
    return output_dict


def prepare_frames(dataset, device, max_frames):
    n = min(len(dataset), max_frames)
    frames = []
    for i in range(n):
        item = dataset[i]
        batch = dataset.collate_batch_test([item])
        batch = train_utils.to_device(batch, device)
        frames.append(batch)
    return frames


@torch.no_grad()
def _timed_forward(batch, model, dataset, fusion_method):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    output_dict = forward_only(batch, model, fusion_method)
    torch.cuda.synchronize()
    after_fwd = time.perf_counter()
    dataset.post_process(batch, output_dict)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (after_fwd - t0) * 1e3, (t2 - after_fwd) * 1e3, (t2 - t0) * 1e3


@torch.no_grad()
def benchmark_model(model, dataset, frames, fusion_method, repeats, warmup, device):
    cold_fwd, cold_post, cold_e2e = _timed_forward(
        frames[0], model, dataset, fusion_method)

    warmup_e2e = []
    for i in range(warmup):
        _, _, e = _timed_forward(frames[i % len(frames)], model,
                                 dataset, fusion_method)
        warmup_e2e.append(e)

    fwd_times = []
    post_times = []
    e2e_times = []
    first_round_e2e = []

    for rep in range(repeats):
        for batch in frames:
            f, p, e = _timed_forward(batch, model, dataset, fusion_method)
            fwd_times.append(f)
            post_times.append(p)
            e2e_times.append(e)
            if rep == 0:
                first_round_e2e.append(e)

    def stat(xs):
        return {
            'mean': statistics.mean(xs),
            'median': statistics.median(xs),
            'std': statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            'min': min(xs),
            'max': max(xs),
        }

    return {
        'forward_ms': stat(fwd_times),
        'post_ms': stat(post_times),
        'e2e_ms': stat(e2e_times),
        'num_timed_frames': len(e2e_times),
        'cold_start': {
            'forward_ms': cold_fwd,
            'post_ms': cold_post,
            'e2e_ms': cold_e2e,
        },
        'warmup': {
            'num_iters': warmup,
            'e2e_ms': warmup_e2e,
            'first_e2e_ms': warmup_e2e[0] if warmup_e2e else None,
            'last_e2e_ms': warmup_e2e[-1] if warmup_e2e else None,
            'mean_e2e_ms': statistics.mean(warmup_e2e) if warmup_e2e else None,
        },
        'first_timed_round_e2e_ms': stat(first_round_e2e) if first_round_e2e else None,
    }


def main():
    args = build_parser().parse_args()

    if not torch.cuda.is_available():
        print('CUDA not available; aborting.', file=sys.stderr)
        sys.exit(2)
    device = torch.device(args.device)

    print(f'[bench] model_dir={args.model_dir}')
    hypes = yaml_utils.load_yaml(os.path.join(args.model_dir, 'config.yaml'), None)

    print('[bench] building dataset ...')
    dataset = build_dataset(hypes, visualize=False, train=False, isSim=False)

    print('[bench] creating model ...')
    model = train_utils.create_model(hypes)
    model.to(device)
    epoch, model = train_utils.load_saved_model(args.model_dir, model)
    model.eval()

    print(f'[bench] preparing up to {args.max_frames} frames (epoch={epoch}) ...')
    frames = prepare_frames(dataset, device, args.max_frames)
    print(f'[bench] {len(frames)} frames resident on GPU')

    result = benchmark_model(model, dataset, frames, args.fusion_method,
                             args.repeats, args.warmup, device)

    cs = result['cold_start']
    wu = result['warmup']
    e2e_final = result['e2e_ms']['mean']
    _log(f'[bench] cold start   e2e={cs["e2e_ms"]:.1f} ms '
         f'(fwd={cs["forward_ms"]:.1f} post={cs["post_ms"]:.1f})')
    if wu['num_iters'] > 0:
        _log(f'[bench] warm-up x{wu["num_iters"]}  '
             f'first={wu["first_e2e_ms"]:.1f} ms  last={wu["last_e2e_ms"]:.1f} ms  '
             f'mean={wu["mean_e2e_ms"]:.1f} ms')
        _log(f'[bench] timed mean e2e={e2e_final:.1f} ms  '
             f'(warm-up saved ~{cs["e2e_ms"] - e2e_final:.1f} ms vs cold)')

    gpu_name = torch.cuda.get_device_name(device)
    payload = {
        'model_dir': args.model_dir,
        'model_name': os.path.basename(os.path.normpath(args.model_dir)),
        'model_core_method': hypes['model']['core_method'],
        'fusion_method': args.fusion_method,
        'dataset_core_method': hypes['fusion']['core_method'],
        'checkpoint_epoch': epoch,
        'num_frames': len(frames),
        'repeats': args.repeats,
        'warmup': warmup_arg(args),
        'device': str(device),
        'gpu_name': gpu_name,
        'result': result,
    }

    print(json.dumps(payload, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'[bench] wrote {args.output_json}')


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def warmup_arg(args):
    return args.warmup


if __name__ == '__main__':
    main()
