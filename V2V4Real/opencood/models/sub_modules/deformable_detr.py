
import torch
import torch.nn.functional as F
from torch import nn
import math

from opencood.models.sub_modules.deformable_transformer import build_deforamble_transformer
import copy
from opencood.models.sub_modules.position_encoding import PositionEmbeddingSine
from einops.layers.torch import Rearrange, Reduce
from opencood.models.sub_modules.mln import MLN, nerf_positional_encoding, AT


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.ego_id = 0
        self.time_id = -1
        self.prev_len = args['prev_size']

        hidden_dim = args['hidden_dim']
        self.hidden_dim = hidden_dim
        self.pos_emb = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.multi_time = args['multi_time']
        self.dif_offset = args['dif_offset'] if 'dif_offset' in args else False
        self.num_layers = args['enc_layers']

        if self.dif_offset:
            self.query_embed = AT(1)

        if self.multi_time:
            max_agents = args['max_cav'] + 1
        else:
            max_agents = args['max_cav']
        self.transformer = build_deforamble_transformer(
            args, num_feature_levels=max_agents, num_encoder_layers=args['enc_layers'],
            dif_offset=False)

        self.mlp_head = nn.Sequential(
            Reduce('b l h w c-> b h w c', 'mean'),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            Rearrange('b h w c -> b c h w')
        )

    def forward(self, src, mask=None, prior_encoding=None):
        B, L, C, H, W = src.size()
        if mask is None:
            mask = torch.ones(B, L).bool().to(src.device)

        if self.dif_offset:
            src = src + self.query_embed(src, prior_encoding)

        src = src.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        mask = mask.reshape(L, B, 1, 1).repeat(1, 1, H, W)
        pos = self.pos_emb(mask.flatten(0, 1)).reshape(L, B, C, H, W)
        mask = ~mask.bool()

        memory, (spatial_shapes, level_start_index) = self.transformer(
            src, mask, pos, prior_encoding=prior_encoding)
        out = self.mlp_head(memory.reshape(B, L, H, W, C))

        return out
