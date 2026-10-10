import random
import math
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.data_utils.datasets
from opencood.utils import box_utils
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2


class EarlyFusionDatasetEgoPreserving(basedataset.BaseDataset):
    def __init__(self, params, visualize, train=True, isSim=False):
        super(EarlyFusionDatasetEgoPreserving, self).__init__(params, visualize, train, isSim)
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = build_postprocessor(params['postprocess'], train)
        self.ego_only = params.get('ego_only', False)
        self.augment_config = params['data_augment']

    def generate_augment(self):
        flip = [None, None, None]
        noise_rotation = None
        noise_scale = None

        for aug_ele in self.augment_config:
            if 'random_world_rotation' in aug_ele['NAME']:
                rot_range = aug_ele['WORLD_ROT_ANGLE']
                if not isinstance(rot_range, list):
                    rot_range = [-rot_range, rot_range]
                noise_rotation = np.random.uniform(rot_range[0],
                                                   rot_range[1])

            if 'random_world_flip' in aug_ele['NAME']:
                for i, cur_axis in enumerate(aug_ele['ALONG_AXIS_LIST']):
                    enable = np.random.choice([False, True], replace=False,
                                              p=[0.5, 0.5])
                    flip[i] = enable

            if 'random_world_scaling' in aug_ele['NAME']:
                scale_range = aug_ele['WORLD_SCALE_RANGE']
                noise_scale = np.random.uniform(scale_range[0],
                                                scale_range[1])

        return flip, noise_rotation, noise_scale

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}
        processed_data_dict['ego']['ego_processed_lidar'] = None

        ego_id = -1
        ego_lidar_pose = []

        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        projected_lidar_stack = []
        ego_projected_lidar_stack = []
        object_stack = []
        object_id_stack = []

        for cav_id, selected_cav_base in base_data_dict.items():
            if self.ego_only and not self.train and not selected_cav_base['ego']:
                continue
            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)
            projected_lidar_stack.append(
                selected_cav_processed['projected_lidar'])
            if selected_cav_base['ego']:
                ego_projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']

        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        projected_lidar_stack = np.vstack(projected_lidar_stack)
        ego_projected_lidar_stack = np.vstack(ego_projected_lidar_stack)

        flip, noise_rotation, noise_scale = self.generate_augment()

        fused_object_bbx_center = object_bbx_center.copy()
        fused_mask = mask.copy()
        projected_lidar_stack, fused_object_bbx_center, fused_mask = \
            self.augment(projected_lidar_stack,
                         fused_object_bbx_center,
                         fused_mask,
                         flip,
                         noise_rotation,
                         noise_scale)
        ego_projected_lidar_stack, _, _ = \
            self.augment(ego_projected_lidar_stack.copy(),
                         object_bbx_center.copy(),
                         mask.copy(),
                         flip,
                         noise_rotation,
                         noise_scale)

        projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                     self.params['preprocess'][
                                                         'cav_lidar_range'])
        ego_projected_lidar_stack = mask_points_by_range(ego_projected_lidar_stack,
                                                         self.params['preprocess'][
                                                             'cav_lidar_range'])
        object_bbx_center_valid = fused_object_bbx_center[fused_mask == 1]
        object_bbx_center_valid, valid_mask = \
            box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
                                                     self.params['preprocess'][
                                                         'cav_lidar_range'],
                                                     self.params[
                                                         'postprocess'][
                                                         'order']
                                                     )
        fused_mask[fused_object_bbx_center.shape[0]:] = 0
        fused_object_bbx_center[:object_bbx_center_valid.shape[0]] = \
            object_bbx_center_valid
        fused_object_bbx_center[object_bbx_center_valid.shape[0]:] = 0
        unique_indices = [unique_indices[i] for i, n in enumerate(list(valid_mask)) if n]

        lidar_dict = self.pre_processor.preprocess(projected_lidar_stack)
        ego_lidar_dict = self.pre_processor.preprocess(ego_projected_lidar_stack)

        anchor_box = self.post_processor.generate_anchor_box()

        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=fused_object_bbx_center,
                anchors=anchor_box,
                mask=fused_mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': fused_object_bbx_center,
             'object_bbx_mask': fused_mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': lidar_dict,
             'ego_processed_lidar': ego_lidar_dict,
             'label_dict': label_dict,
             'folder_name': base_data_dict[ego_id]['folder_name'],
             'timestamp_index': base_data_dict[ego_id]['index'],
             'timestamp_key': base_data_dict[ego_id]['timestamp_key'],
             'scenario_index': base_data_dict[ego_id]['scenario_index'],
             'cav_id': base_data_dict[ego_id]['cav_id']})

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                                                   projected_lidar_stack,
                                               'ego_origin_lidar':
                                                   base_data_dict[ego_id]['lidar_np']})

        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose):
        selected_cav_processed = {}

        transformation_matrix = selected_cav_base['params'][
            'transformation_matrix']

        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       transformation_matrix if not self.isSim else ego_pose)

        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np)
        lidar_np[:, :3] = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                     transformation_matrix)

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np})

        return selected_cav_processed

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        output_dict = {}

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            object_bbx_center = \
                torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = \
                torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']

            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box':
                    torch.from_numpy(np.array(
                        cav_content[
                            'anchor_box']))})
            if self.visualize:
                origin_lidar = [cav_content['origin_lidar']]

            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(
                    [cav_content['processed_lidar']])
            ego_processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(
                    [cav_content['ego_processed_lidar']])
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']])

            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()

            output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'processed_lidar': processed_lidar_torch_dict,
                                        'ego_processed_lidar': ego_processed_lidar_torch_dict,
                                        'label_dict': label_torch_dict,
                                        'object_ids': object_ids,
                                        'transformation_matrix': transformation_matrix_torch,
                                        'folder_name': cav_content['folder_name'],
                                        'timestamp_index': cav_content['timestamp_index'],
                                        'timestamp_key': cav_content['timestamp_key'],
                                        'scenario_index': cav_content['scenario_index'],
                                        'cav_id': cav_content['cav_id']})

            if self.visualize:
                origin_lidar = \
                    np.array(
                        downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                ego_origin_lidar = cav_content['ego_origin_lidar']
                ego_origin_lidar = np.array(
                    downsample_lidar_minimum(pcd_np_list=[ego_origin_lidar]))
                ego_origin_lidar = torch.from_numpy(ego_origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar,
                                            'ego_origin_lidar': ego_origin_lidar})

        return output_dict

    def collate_batch_train(self, batch):
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        ego_processed_lidar_list = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            ego_processed_lidar_list.append(ego_dict['ego_processed_lidar'])
            label_dict_list.append(ego_dict['label_dict'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(processed_lidar_list)
        ego_processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(ego_processed_lidar_list)
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)

        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'ego_processed_lidar': ego_processed_lidar_torch_dict,
                                   'label_dict': label_torch_dict})

        if self.visualize:
            origin_lidar = np.array(origin_lidar)
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def post_process(self, data_dict, output_dict):
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor


if __name__ == '__main__':
    params = load_yaml('../../hypes_yaml/voxelnet_early_fusion.yaml')

    opencda_dataset = EarlyFusionDatasetEgoPreserving(params, train=True, visualize=True)
    opencda_dataset.__getitem__(10)
