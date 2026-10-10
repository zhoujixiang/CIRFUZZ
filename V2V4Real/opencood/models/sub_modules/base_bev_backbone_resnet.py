
import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.resblock import ResNetModified, BasicBlock


class ResNetBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels=64):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = model_cfg.get('layer_nums', [])
        layer_strides = model_cfg.get('layer_strides', [])
        num_filters = model_cfg.get('num_filters', [])
        upsample_strides = model_cfg.get('upsample_strides', [])
        num_upsample_filters = model_cfg.get('num_upsample_filter', [])

        self.resnet = ResNetModified(
            BasicBlock,
            layer_nums,
            layer_strides,
            num_filters,
            inplanes=model_cfg.get('inplanes', 64)
        )

        num_levels = len(layer_nums)
        self.num_levels = num_levels
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride, stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3,
                                       momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        x = self.resnet(spatial_features)
        ups = []
        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        data_dict['spatial_features_2d'] = x
        return data_dict

    def get_multiscale_feature(self, spatial_features):
        x = self.resnet(spatial_features)
        return x

    def decode_multiscale_feature(self, x):
        ups = []
        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        return x
