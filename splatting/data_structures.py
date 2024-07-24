# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: MIT License

"""
Adapted from https://github.com/graphdeco-inria/gaussian-splatting
"""

from typing import NamedTuple
import numpy as np


class CameraData(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    params: np.array
    fov_y: np.array
    fov_x: np.array
    c_x: np.array
    c_y: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class PointCloudData(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class SceneData(NamedTuple):
    point_cloud: PointCloudData
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    dimensions: int

