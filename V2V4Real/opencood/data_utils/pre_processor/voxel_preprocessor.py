import sys

import numpy as np
import torch

from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor


class VoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(VoxelPreprocessor, self).__init__(preprocess_params, train)
        self.lidar_range = self.params['cav_lidar_range']

        self.vw = self.params['args']['vw']
        self.vh = self.params['args']['vh']
        self.vd = self.params['args']['vd']
        self.T = self.params['args']['T']

    def preprocess(self, pcd_np):
        data_dict = {}

        voxel_coords = ((pcd_np[:, :3] -
                         np.floor(np.array([self.lidar_range[0],
                                            self.lidar_range[1],
                                            self.lidar_range[2]])) / (
                             self.vw, self.vh, self.vd))).astype(np.int32)

        voxel_coords = voxel_coords[:, [2, 1, 0]]
        voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0,
                                                        return_inverse=True,
                                                        return_counts=True)

        voxel_features = []

        for i in range(len(voxel_coords)):
            voxel = np.zeros((self.T, 7), dtype=np.float32)
            pts = pcd_np[inv_ind == i]
            if voxel_counts[i] > self.T:
                pts = pts[:self.T, :]
                voxel_counts[i] = self.T

            voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] -
                                                      np.mean(pts[:, :3], 0)),
                                                     axis=1)
            voxel_features.append(voxel)

        data_dict['voxel_features'] = np.array(voxel_features)
        data_dict['voxel_coords'] = voxel_coords

        return data_dict

    def collate_batch(self, batch):

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def collate_batch_list(batch):
        voxel_features = []
        voxel_coords = []

        for i in range(len(batch)):
            voxel_features.append(batch[i]['voxel_features'])
            coords = batch[i]['voxel_coords']
            voxel_coords.append(
                np.pad(coords, ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))

        voxel_features = torch.from_numpy(np.concatenate(voxel_features))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords}

    @staticmethod
    def collate_batch_dict(batch: dict):
        voxel_features = \
            torch.from_numpy(np.concatenate(batch['voxel_features']))
        coords = batch['voxel_coords']
        voxel_coords = []

        for i in range(len(coords)):
            voxel_coords.append(
                np.pad(coords[i], ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords}
