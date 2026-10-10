import numpy as np
from collections import OrderedDict

from opencood.data_utils.datasets.intermediate_fusion_dataset import \
    IntermediateFusionDataset
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils import box_utils
from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points, \
    shuffle_points
from opencood.utils.transformation_utils import x1_to_x2


class IntermediateFusionDatasetWarp(IntermediateFusionDataset):

    def __init__(self, params, visualize, train=True, isSim=False):
        super(IntermediateFusionDatasetWarp, self). \
            __init__(params, visualize, train, isSim)
        self.params['data_augment'] = []
        self.data_augmentor = DataAugmentor([], train, intermediate=True)

    def get_item_single_car(self, selected_cav_base, ego_pose):
        selected_cav_processed = {}

        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']

        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center(
                [selected_cav_base],
                transformation_matrix if not self.isSim else ego_pose)

        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np)
        lidar_np, object_bbx_center, object_bbx_mask = \
            self.augment(lidar_np, object_bbx_center, object_bbx_mask,
                         selected_cav_base['flip'],
                         selected_cav_base['noise_rotation'],
                         selected_cav_base['noise_scale'])
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        void_lidar = True if lidar_np.shape[0] < 1 else False

        object_bbx_center_valid = object_bbx_center[object_bbx_mask == 1]
        object_bbx_center_valid, range_mask = \
            box_utils.mask_boxes_outside_range_numpy(
                object_bbx_center_valid,
                self.params['preprocess']['cav_lidar_range'],
                self.params['postprocess']['order'])
        object_ids = \
            [int(x) for x in list(np.array(object_ids)[range_mask])]
        processed_lidar = self.pre_processor.preprocess(lidar_np)

        velocity = selected_cav_base['params']['ego_speed']
        velocity = velocity / 30

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center_valid,
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'processed_features': processed_lidar,
             'velocity': velocity})

        return selected_cav_processed, void_lidar

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1))

        pose_list = [cav_content['params']['lidar_pose']
                     for cav_content in base_data_dict.values()]

        for i in range(len(pose_list)):
            for j in range(len(pose_list)):
                if i != j:
                    pairwise_t_matrix[i, j] = \
                        x1_to_x2(pose_list[i], pose_list[j])

        return pairwise_t_matrix
