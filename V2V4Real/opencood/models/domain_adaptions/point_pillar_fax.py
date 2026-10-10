import torch
import torch.nn as nn
from einops import rearrange, repeat

from opencood.models.point_pillar_fax import \
    PointPillarFax as PointPillarFaxBase, regroup


class PointPillarFax(PointPillarFaxBase):

    def forward(self, data_dict):
        if not self.training:
            return super().forward(data_dict)

        feature = []
        for data_per in data_dict:
            voxel_features = data_per['processed_lidar']['voxel_features']
            voxel_coords = data_per['processed_lidar']['voxel_coords']
            voxel_num_points = data_per['processed_lidar']['voxel_num_points']
            record_len = data_per['record_len']
            spatial_correction_matrix = data_per['spatial_correction_matrix']

            batch_dict = {'voxel_features': voxel_features,
                          'voxel_coords': voxel_coords,
                          'voxel_num_points': voxel_num_points,
                          'record_len': record_len}
            batch_dict = self.pillar_vfe(batch_dict)
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)

            spatial_features_2d = batch_dict['spatial_features_2d']
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            if self.compression:
                spatial_features_2d = self.naive_compressor(
                    spatial_features_2d)

            regroup_feature, mask = regroup(spatial_features_2d,
                                            record_len,
                                            self.max_cav)
            com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            com_mask = repeat(com_mask,
                              'b h w c l -> b (h new_h) (w new_w) c l',
                              new_h=regroup_feature.shape[3],
                              new_w=regroup_feature.shape[4])

            fused_feature = self.fusion_net(regroup_feature, com_mask)
            feature.append(fused_feature)

        output_dict = {'psm': self.cls_head(feature[0]),
                       'rm': self.reg_head(feature[0]),
                       'source_feature': feature[0],
                       'target_feature': feature[1],
                       'target_psm': self.cls_head(feature[1]),
                       'target_rm': self.reg_head(feature[1])
                       }

        return output_dict
