#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
Adapted from https://github.com/graphdeco-inria/gaussian-splatting/utils/camera_utils.py
"""

from splatting.splatting_utils import *

def load_cam(id, camera_data, resolution_scale, device='cuda'):
    orig_w, orig_h = camera_data.image.size

    global_down = 1

    if orig_w > 1600:
        global_down = orig_w / 1600

    scale = float(global_down) * float(resolution_scale)
    resolution = (int(orig_w / scale), int(orig_h / scale))

    c_x = (camera_data.c_x + 0.5) / scale - 0.5
    c_y = (camera_data.c_y + 0.5) / scale - 0.5

    c_x = (c_x - resolution[0] / 2) / resolution[0] * 2
    c_y = (c_y - resolution[1] / 2) / resolution[1] * 2

    resized_image_rgb = PIL_to_torch(camera_data.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=camera_data.uid, R=camera_data.R, T=camera_data.T,
                  fov_x=camera_data.fov_x, fov_y=camera_data.fov_y, c_x=c_x, c_y=c_y,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=camera_data.image_name, params=camera_data.params, uid=id, device=device)


def cameras_list_from_cam_data(cams, resolution_scale):
    camera_list = []

    for id, c in enumerate(cams):
        camera_list.append(load_cam(id, c, resolution_scale))

    return camera_list


class Camera:
    def __init__(self, colmap_id, R, T, fov_x, fov_y, c_x, c_y, image, gt_alpha_mask,
                 image_name, uid, params=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, device="cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.c_x = c_x
        self.c_y = c_y
        self.image_name = image_name
        self.params = params
        self.device = device

        self.original_image = image.clamp(0.0, 1.0).to(self.device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(get_world_to_view_alt(R, T, trans, scale)).transpose(0, 1).to(self.device)
        self.projection_matrix = get_projection_matrix(znear=self.znear, zfar=self.zfar, fovX=self.fov_x, fovY=self.fov_y, cX=self.c_x, cY=self.c_y, W=self.image_width, H=self.image_height).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]