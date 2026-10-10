import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import common_utils


def test_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', type=str, required=True)
    p.add_argument('--frames', type=int, default=5)
    return p.parse_args()


def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)
    hypes_off = dict(hypes)
    hypes_off['postprocess'] = dict(hypes['postprocess'])
    hypes_off['postprocess'].pop('ego_protection', None)

    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    from opencood.data_utils.datasets import LateFusionDataset
    ds_on = LateFusionDataset(hypes, visualize=True, train=False)
    ds_off = LateFusionDataset(hypes_off, visualize=True, train=False)
    dl_on = DataLoader(ds_on, batch_size=1, num_workers=8,
                       collate_fn=ds_on.collate_batch_test, shuffle=False)
    dl_off = DataLoader(ds_off, batch_size=1, num_workers=8,
                        collate_fn=ds_off.collate_batch_test, shuffle=False)

    total_on = 0
    total_off = 0
    frames = 0
    for (b_on, b_off) in zip(dl_on, dl_off):
        with torch.no_grad():
            b_on = train_utils.to_device(b_on, device)
            b_off = train_utils.to_device(b_off, device)
            out_on = {k: model(c) for k, c in b_on.items() if k in b_on}
            out_off = {k: model(c) for k, c in b_off.items() if k in b_off}
            pb_on, sc_on = ds_on.post_processor.post_process(b_on, out_on)
            pb_off, sc_off = ds_off.post_processor.post_process(b_off, out_off)

        n_on = 0 if pb_on is None else pb_on.shape[0]
        n_off = 0 if pb_off is None else pb_off.shape[0]
        total_on += n_on
        total_off += n_off
        frames += 1
        if frames % 5 == 1 or n_on != n_off:
            print('frame %d: off=%d on=%d' % (frames - 1, n_off, n_on))
        if frames >= opt.frames:
            break

    print('\nSUMMARY over %d frames:' % frames)
    print('  merged boxes (protection OFF): %d' % total_off)
    print('  merged boxes (protection ON) : %d' % total_on)


if __name__ == '__main__':
    main()
