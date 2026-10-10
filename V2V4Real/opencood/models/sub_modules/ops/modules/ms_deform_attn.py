
from __future__ import absolute_import, print_function, division

import warnings
from typing import Optional

import torch
from torch import nn

try:
    from mmcv.ops import MultiScaleDeformableAttention as MMCV_MSDA
except Exception as e:
    raise ImportError(
        "Failed to import mmcv.ops.MultiScaleDeformableAttention. "
        "Please install/compile MMCV with CUDA ops (MMCV_WITH_OPS=1). "
        f"Original error: {e}"
    )

__all__ = ["MSDeformAttn"]


def _is_power_of_2(n: int) -> bool:
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        dif_offset: bool = False,
        cav_num: int = 2,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}"
            )
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model so that each head dim is a power of 2 "
                "(more efficient in CUDA implementation)."
            )

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.cav_num = cav_num
        self.dif_offset = dif_offset

        self._batch_first = True
        try:
            self.msda = MMCV_MSDA(
                embed_dims=d_model,
                num_levels=n_levels,
                num_heads=n_heads,
                num_points=n_points,
                batch_first=True,
            )
        except TypeError:
            self.msda = MMCV_MSDA(
                embed_dims=d_model,
                num_levels=n_levels,
                num_heads=n_heads,
                num_points=n_points,
            )
            self._batch_first = False

        if self.dif_offset:
            try:
                self.msda_infra = MMCV_MSDA(
                    embed_dims=d_model,
                    num_levels=n_levels,
                    num_heads=n_heads,
                    num_points=n_points,
                    batch_first=True,
                )
            except TypeError:
                self.msda_infra = MMCV_MSDA(
                    embed_dims=d_model,
                    num_levels=n_levels,
                    num_heads=n_heads,
                    num_points=n_points,
                )

    def _reset_parameters(self):
        for m in [self.msda, getattr(self, "msda_infra", None)]:
            if m is None:
                continue
            if hasattr(m, "init_weights"):
                m.init_weights()
            elif hasattr(m, "_reset_parameters"):
                m._reset_parameters()
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def _call_mmcv(
        self,
        module: MMCV_MSDA,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        need_transpose = not self._batch_first
        if need_transpose:
            query_t = query.transpose(0, 1).contiguous()
            value_t = value.transpose(0, 1).contiguous()
            out_t = module(
                query=query_t,
                reference_points=reference_points,
                value=value_t,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                identity=torch.zeros_like(query_t),
            )
            return out_t.transpose(0, 1).contiguous()

        return module(
            query=query,
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
            identity=torch.zeros_like(query),
        )

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
        types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_spatial_shapes.numel() > 0:
            expected_len = (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum().item()
            if input_flatten.size(1) != expected_len:
                raise AssertionError(
                    f"input_flatten length mismatch: got {input_flatten.size(1)}, "
                    f"expected {expected_len} from spatial_shapes"
                )

        if self.dif_offset and types is not None:
            out0 = self._call_mmcv(
                self.msda,
                query=query,
                reference_points=reference_points,
                value=input_flatten,
                spatial_shapes=input_spatial_shapes,
                level_start_index=input_level_start_index,
                key_padding_mask=input_padding_mask,
            )
            out1 = self._call_mmcv(
                self.msda_infra,
                query=query,
                reference_points=reference_points,
                value=input_flatten,
                spatial_shapes=input_spatial_shapes,
                level_start_index=input_level_start_index,
                key_padding_mask=input_padding_mask,
            )
            mask = (types == 0)
            while mask.dim() < out0.dim():
                mask = mask.unsqueeze(-1)
            mask = mask.to(dtype=torch.bool, device=out0.device)
            return torch.where(mask, out0, out1)

        return self._call_mmcv(
            self.msda,
            query=query,
            reference_points=reference_points,
            value=input_flatten,
            spatial_shapes=input_spatial_shapes,
            level_start_index=input_level_start_index,
            key_padding_mask=input_padding_mask,
        )
