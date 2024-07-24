# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: MIT License

"""
Adapted from https://github.com/graphdeco-inria/gaussian-splatting
"""

import os
import random

from splatting.dataset_readers.colmap import read_colmap_scene
from splatting.dataset_readers.blender import readNerfSyntheticInfo
from splatting.camera import *


class Scene:

    def __init__(self, dataset_path, shuffle=True, white_background=False, resolution_scales=[1.0]):

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(dataset_path, "transforms_train.json")):
            print("Reading Blender Scene")
            scene_data = readNerfSyntheticInfo(dataset_path, white_background=white_background, eval=False)
        elif os.path.exists(os.path.join(dataset_path, "sparse")):
            print("Reading COLMAP Scene")
            scene_data = read_colmap_scene(dataset_path, eval=True)
        else:
            assert False, "Scene type not supported"

        self.data = scene_data

        # Read cameras
        cam_list = []

        if scene_data.test_cameras:
            cam_list.extend(scene_data.test_cameras)
        if scene_data.train_cameras:
            cam_list.extend(scene_data.train_cameras)

        if shuffle:
            random.shuffle(scene_data.train_cameras)

        self.cameras_extent = scene_data.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras = cameras_list_from_cam_data(scene_data.train_cameras, resolution_scale)
            print("Loading Test Cameras")
            self.test_cameras = cameras_list_from_cam_data(scene_data.test_cameras, resolution_scale)

    def total_parameters(self):
        return self.train_cameras[0].params.shape[0]
