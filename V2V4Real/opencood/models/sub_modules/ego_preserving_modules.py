import torch
import torch.nn as nn
from einops import repeat

from opencood.models.mwin_tranformer import V2XTransformer
from opencood.models.sub_modules.f_cooper_fuse import SpatialFusion
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.self_attn import AttFusion
from opencood.models.sub_modules.swap_fusion_modules import SwapFusionEncoder
from opencood.models.sub_modules.v2v_fuse import V2VNetFusion


def infer_feature_dim(args):
    ego_args = args.get('ego_preserving', {})
    if 'feature_dim' in ego_args:
        return ego_args['feature_dim']

    if 'shrink_header' in args and args['shrink_header']['dim']:
        return args['shrink_header']['dim'][-1]

    return 256


def unpad_prior_encoding(prior_encoding, record_len):
    out = []
    for i in range(prior_encoding.shape[0]):
        out.append(prior_encoding[i, :record_len[i], :])
    return torch.cat(out, dim=0)


class EgoPreservingHead(nn.Module):

    def __init__(self, feature_dim, anchor_number):
        super(EgoPreservingHead, self).__init__()
        self.feature_dim = feature_dim

        self.gate_net = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, kernel_size=1, bias=True)
        )

        self.cls_head = nn.Conv2d(feature_dim,
                                  anchor_number,
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(feature_dim,
                                  7 * anchor_number,
                                  kernel_size=1)

        self.ego_cls_head = nn.Conv2d(feature_dim,
                                      anchor_number,
                                      kernel_size=1)
        self.ego_reg_head = nn.Conv2d(feature_dim,
                                      7 * anchor_number,
                                      kernel_size=1)

    def freeze_detection_heads(self):
        for p in self.cls_head.parameters():
            p.requires_grad = False

        for p in self.reg_head.parameters():
            p.requires_grad = False

        for p in self.ego_cls_head.parameters():
            p.requires_grad = False

        for p in self.ego_reg_head.parameters():
            p.requires_grad = False

    def forward(self, ego_feature, cooperative_feature):
        cooperative_delta = cooperative_feature - ego_feature
        gate_input = torch.cat([ego_feature,
                                cooperative_feature,
                                cooperative_delta], dim=1)
        fusion_gate = torch.sigmoid(self.gate_net(gate_input))
        protected_feature = ego_feature + fusion_gate * cooperative_delta

        return {
            'psm': self.cls_head(protected_feature),
            'rm': self.reg_head(protected_feature),
            'ego_psm': self.ego_cls_head(ego_feature),
            'ego_rm': self.ego_reg_head(ego_feature),
            'fusion_gate': fusion_gate
        }


class CooperativeFusionAdapter(nn.Module):

    def __init__(self, args, feature_dim):
        super(CooperativeFusionAdapter, self).__init__()

        ego_args = args.get('ego_preserving', {})
        self.fusion_method = ego_args.get('fusion_method', 'v2vnet')
        self.max_cav = args['max_cav']
        self.feature_dim = feature_dim

        if self.fusion_method == 'v2vnet':
            self.fusion_net = V2VNetFusion(args['v2vfusion'])
        elif self.fusion_method == 'fcooper':
            self.fusion_net = SpatialFusion()
        elif self.fusion_method == 'opv2v':
            self.fusion_net = AttFusion(feature_dim)
        elif self.fusion_method == 'transformer':
            self.fusion_net = V2XTransformer(args['transformer'])
        elif self.fusion_method == 'fax':
            self.fusion_net = SwapFusionEncoder(args['fax_fusion'])
        else:
            raise ValueError('Unsupported fusion_method: %s' %
                             self.fusion_method)

    def forward(self, spatial_features_2d, data_dict):
        record_len = data_dict['record_len']
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        ego_feature = regroup_feature[:, 0, ...]

        if self.fusion_method == 'v2vnet':
            prior_encoding = unpad_prior_encoding(data_dict['prior_encoding'],
                                                  record_len)
            cooperative_feature = self.fusion_net(
                spatial_features_2d,
                record_len,
                data_dict['pairwise_t_matrix'],
                prior_encoding
            )
        elif self.fusion_method in ['fcooper', 'opv2v']:
            cooperative_feature = self.fusion_net(spatial_features_2d,
                                                  record_len)
        elif self.fusion_method == 'transformer':
            prior_encoding = data_dict['prior_encoding'].unsqueeze(-1) \
                .unsqueeze(-1)
            prior_encoding = prior_encoding.repeat(1,
                                                   1,
                                                   1,
                                                   regroup_feature.shape[3],
                                                   regroup_feature.shape[4])
            transformer_input = torch.cat([regroup_feature, prior_encoding],
                                          dim=2)
            transformer_input = transformer_input.permute(0, 1, 3, 4, 2)
            cooperative_feature = self.fusion_net(
                transformer_input,
                mask,
                data_dict['spatial_correction_matrix']
            )
            cooperative_feature = cooperative_feature.permute(0, 3, 1, 2)
        else:
            com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            com_mask = repeat(com_mask,
                              'b h w c l -> b (h new_h) (w new_w) c l',
                              new_h=regroup_feature.shape[3],
                              new_w=regroup_feature.shape[4])
            cooperative_feature = self.fusion_net(regroup_feature, com_mask)

        return ego_feature, cooperative_feature
