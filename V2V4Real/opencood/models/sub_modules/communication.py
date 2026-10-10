import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange


class CompressFuse(nn.Module):
    def __init__(self, args, max_cav=5):
        super(CompressFuse, self).__init__()
        L = max_cav
        self.ego_id = 0

        self.smooth = False

        self.rescale_rate = args['rescale_rate'] if 'rescale_rate' in args else 10
        self.thresh = args['thresh'] if 'thresh' in args else 0.01
        print('thresh:', self.thresh)
        self.reconstruct = args['reconstruct'] if 'reconstruct' in args else False

        if self.reconstruct:
            input_dim = args['mlp_dim']
            if self.reconstruct == 'conv':
                self.conv_fuse = nn.Sequential(
                    Rearrange('b l c h w -> (b l) c h w', l=L),
                    nn.Conv2d(2 * input_dim, input_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(input_dim),
                    nn.GELU(),
                    Rearrange('(b l) c h w -> b l c h w', l=L),
                )
            else:
                raise NotImplementedError

            hidden_dim = input_dim // 4
            self.use_decoder = 'use_decoder' in args
            if self.use_decoder is False:
                self.decoder = nn.Identity()
            else:
                self.decoder = nn.Sequential(
                    Rearrange('b l c h w -> (b l) c h w', l=L),
                    nn.Conv2d(input_dim, hidden_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, hidden_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, input_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(input_dim),
                    nn.ReLU(),
                    Rearrange('(b l) c h w -> b l c h w', l=L),
                )

    def forward(self, feat, confidence_map, prev_feat=None, pre_map=None, reset=False):
        B, L, _, H, W = confidence_map.shape

        communication_maps = confidence_map

        ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
        zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)

        if reset:
            dynamic_communication_maps = torch.where(
                communication_maps > self.thresh, ones_mask, zeros_mask)
        else:
            dynamic_map = torch.abs(communication_maps - pre_map)
            dynamic_communication_maps = communication_maps * (1 / (self.rescale_rate + 1) + dynamic_map * self.rescale_rate)
            dynamic_communication_maps = torch.where(
                dynamic_communication_maps > self.thresh, ones_mask, zeros_mask)

        dynamic_communication_maps[:, self.ego_id, ...] = 1

        cur_feat = feat * dynamic_communication_maps
        if self.reconstruct:
            if self.use_decoder:
                cur_feat = cur_feat + self.decoder(cur_feat)
            if self.reconstruct == 'conv' and not reset:
                cur_feat = cur_feat + \
                    self.conv_fuse(torch.cat((cur_feat, prev_feat), dim=-3))

        communication_rate = (dynamic_communication_maps[:, self.ego_id+1:,].sum() / (H * W) / B).item()

        return cur_feat, communication_rate
