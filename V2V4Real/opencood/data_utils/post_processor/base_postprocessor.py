
import numpy as np
import torch

from opencood.utils import box_utils


class BasePostprocessor(object):

    def __init__(self, anchor_params, train=True):
        self.params = anchor_params
        self.bbx_dict = {}
        self.train = train

    def generate_anchor_box(self):
        return None

    def generate_label(self, *argv):
        return None

    def generate_gt_bbx(self, data_dict):
        gt_box3d_list = []
        object_id_list = []

        for cav_id, cav_content in data_dict.items():
            transformation_matrix = cav_content['transformation_matrix'] \
                if 'gt_transformation_matrix' not in cav_content \
                else cav_content['gt_transformation_matrix']

            object_bbx_center = cav_content['object_bbx_center']
            object_bbx_mask = cav_content['object_bbx_mask']
            object_ids = cav_content['object_ids']
            object_bbx_center = object_bbx_center[object_bbx_mask == 1]

            object_bbx_corner = \
                box_utils.boxes_to_corners_3d(object_bbx_center,
                                              self.params['order'])
            projected_object_bbx_corner = \
                box_utils.project_box3d(object_bbx_corner.float(),
                                        transformation_matrix)
            gt_box3d_list.append(projected_object_bbx_corner)

            object_id_list += object_ids

        gt_box3d_list = torch.vstack(gt_box3d_list)
        gt_box3d_selected_indices = \
            [object_id_list.index(x) for x in set(object_id_list)]
        gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]

        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor)
        gt_box3d_tensor = gt_box3d_tensor[mask, :, :]

        return gt_box3d_tensor

    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):
        from opencood.data_utils.datasets import GT_RANGE

        tmp_object_dict = {}
        for cav_content in cav_contents:
            tmp_object_dict.update(cav_content['params']['vehicles'])
            cav_id = cav_content['cav_id']

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range'] \
            if self.train else GT_RANGE

        box_utils.project_world_objects(tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'])

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_content) in enumerate(output_dict.items()):
            object_bbx = object_content['coord']
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            if object_content['ass_id'] != -1:
                object_ids.append(object_content['ass_id'])
            else:
                object_ids.append(object_id + 100 * cav_id)

        return object_np, mask, object_ids
