# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: MIT License

import torch
import math


def slice_gaussian(m, q, v, c_dim):
    v_11 = v[:, :c_dim, :c_dim]
    v_12 = v[:, :c_dim, c_dim:]
    v_21 = v[:, c_dim:, :c_dim]
    v_22 = v[:, c_dim:, c_dim:]

    v_22_inv = v_22.inverse()

    v_regr = torch.bmm(v_12, v_22_inv)

    m_1 = m[:, :c_dim]
    m_2 = m[:, c_dim:]
    x = q-m_2

    pdf_cond = torch.exp(-torch.abs(torch.bmm(torch.bmm(x.unsqueeze(-2), v_22_inv), x.unsqueeze(-1))))[:, :, 0]

    m_cond = m_1 + torch.bmm(v_regr, x.unsqueeze(-1))[:, :, 0]
    v_cond = v_11 - torch.bmm(v_regr, v_21)

    return m_cond, v_cond, pdf_cond


"""
Rest is adapted from https://github.com/graphdeco-inria/gaussian-splatting
"""


from splatting.data_structures import *
from plyfile import PlyData, PlyElement


def fetch_ply(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return PointCloudData(points=positions, colors=colors, normals=normals)


def store_ply(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def PIL_to_torch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_nerf_normalization(cam_info):

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = get_world_to_view_alt(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def strip_lower_diag(L):
    elements = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    elements[:, 0] = torch.abs(L[:, 0, 0])
    elements[:, 1] = L[:, 0, 1]
    elements[:, 2] = L[:, 0, 2]
    elements[:, 3] = torch.abs(L[:, 1, 1])
    elements[:, 4] = L[:, 1, 2]
    elements[:, 5] = torch.abs(L[:, 2, 2])
    return elements


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def get_world_to_view(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def get_world_to_view_alt(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def get_projection_matrix(znear, zfar, fovX, fovY, cX, cY, W, H):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = cX
    P[1, 2] = cY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov_to_focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal_to_fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))