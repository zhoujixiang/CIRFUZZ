#!/usr/bin/env python3
import time

import numpy as np
import math
import random
from scipy.spatial.transform import Rotation as R

class LaserScan:
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0,DA=False,flip_sign=False,rot=False,drop_points=False):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.DA = DA
        self.flip_sign = flip_sign
        self.rot = rot
        self.drop_points = drop_points

        self.reset()

    def reset(self):
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.remissions = np.zeros((0, 1), dtype=np.float32)

        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        self.proj_x = np.zeros((0, 1), dtype=np.int32)
        self.proj_y = np.zeros((0, 1), dtype=np.int32)

        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)

    def size(self):
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        self.reset()

        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        points = scan[:, 0:3]
        remissions = scan[:, 3]
        if self.drop_points is not False:
            self.points_to_drop = np.random.randint(0, len(points)-1,int(len(points)*self.drop_points))
            points = np.delete(points,self.points_to_drop,axis=0)
            remissions = np.delete(remissions,self.points_to_drop)

        self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        self.reset()

        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        self.points = points
        if self.flip_sign:
            self.points[:, 1] = -self.points[:, 1]
        if self.DA:
            jitter_x = random.uniform(-5,5)
            jitter_y = random.uniform(-3, 3)
            jitter_z = random.uniform(-1, 0)
            self.points[:, 0] += jitter_x
            self.points[:, 1] += jitter_y
            self.points[:, 2] += jitter_z
        if self.rot:
            self.points = self.points @ R.random(random_state=1234).as_dcm().T
        if remissions is not None:
            self.remissions = remissions
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        if self.project:
            self.do_range_projection()

    def do_range_projection(self):
        fov_up = self.proj_fov_up / 180.0 * np.pi
        fov_down = self.proj_fov_down / 180.0 * np.pi
        fov = abs(fov_down) + abs(fov_up)

        depth = np.linalg.norm(self.points, 2, axis=1)

        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov

        proj_x *= self.proj_W
        proj_y *= self.proj_H

        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)
        self.proj_x = np.copy(proj_x)

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)
        self.proj_y = np.copy(proj_y)

        self.unproj_range = np.copy(depth)

        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)


class SemLaserScan(LaserScan):
    EXTENSIONS_LABEL = ['.label']

    def __init__(self, sem_color_dict=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, max_classes=300,DA=False,flip_sign=False,drop_points=False):
        super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down,DA=DA,flip_sign=flip_sign,drop_points=drop_points)
        self.reset()

        if sem_color_dict:
            max_sem_key = 0
            for key, data in sem_color_dict.items():
                if key + 1 > max_sem_key:
                    max_sem_key = key + 1
            self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
            for key, value in sem_color_dict.items():
                self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
        else:
            max_sem_key = max_classes
            self.sem_color_lut = np.random.uniform(low=0.0,
                                                   high=1.0,
                                                   size=(max_sem_key, 3))
            self.sem_color_lut[0] = np.full((3), 0.1)

        max_inst_id = 100000
        self.inst_color_lut = np.random.uniform(low=0.0,
                                                high=1.0,
                                                size=(max_inst_id, 3))
        self.inst_color_lut[0] = np.full((3), 0.1)

    def reset(self):
        super(SemLaserScan, self).reset()

        self.sem_label = np.zeros((0, 1), dtype=np.int32)
        self.sem_label_color = np.zeros((0, 3), dtype=np.float32)

        self.inst_label = np.zeros((0, 1), dtype=np.int32)
        self.inst_label_color = np.zeros((0, 3), dtype=np.float32)

        self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                       dtype=np.int32)
        self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                       dtype=np.float)

        self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                        dtype=np.int32)
        self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                        dtype=np.float)

    def open_label(self, filename):
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
            raise RuntimeError("Filename extension is not valid label file.")

        label = np.fromfile(filename, dtype=np.int32)
        label = label.reshape((-1))

        if self.drop_points is not False:
            label = np.delete(label,self.points_to_drop)
        self.set_label(label)

    def set_label(self, label):
        if not isinstance(label, np.ndarray):
            raise TypeError("Label should be numpy array")

        if label.shape[0] == self.points.shape[0]:
            self.sem_label = label & 0xFFFF
            self.inst_label = label >> 16
        else:
            print("Points shape: ", self.points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        assert ((self.sem_label + (self.inst_label << 16) == label).all())

        if self.project:
            self.do_label_projection()

    def colorize(self):
        self.sem_label_color = self.sem_color_lut[self.sem_label]
        self.sem_label_color = self.sem_label_color.reshape((-1, 3))

        self.inst_label_color = self.inst_color_lut[self.inst_label]
        self.inst_label_color = self.inst_label_color.reshape((-1, 3))

    def do_label_projection(self):
        mask = self.proj_idx >= 0

        self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
        self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

        self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
        self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]
