# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: MIT License

"""
Adapted from https://github.com/graphdeco-inria/gaussian-splatting
"""

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from splatting_utils import *


def render(camera, model, bg_color, scaling_modifier=1.0, view=0):
    m, v, v_inv, ws, fs, cs, ts = model.get_gs_total(invert=False)

    m_xyz = m[:, 0:3]

    colors = fs
    opacities = ws

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(m_xyz, dtype=m_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(camera.fov_x * 0.5)
    tanfovy = math.tan(camera.fov_y * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=0,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False
    )

    n_gs_init = m.shape[0]

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    cond_params = []

    # View direction
    camera_dir = (m_xyz.detach() - camera.camera_center.repeat(m_xyz.shape[0], 1))
    camera_dir = camera_dir / camera_dir.norm(dim=1, keepdim=True)

    cond_params.append(camera_dir)

    cond_params = torch.cat(cond_params, dim=-1)

    # Cull based on view direction
    mask = model.cull(torch.cat([m_xyz, cond_params], dim=-1), m, v).type(torch.bool)

    screenspace_points = screenspace_points[mask]
    cond_params = cond_params[mask]
    m = m[mask]
    v = v[mask]
    colors = colors[mask]
    opacities = opacities[mask]
    cs = cs[mask]
    ts = ts[mask]

    cull_percent = (n_gs_init-m.shape[0])/n_gs_init

    m_cond, v_cond, pdf_cond = slice_gaussian(m, cond_params, v, c_dim=3)

    opacities = opacities * pdf_cond
    cov = strip_lower_diag(v_cond)

    if view == 1:
        colors = colors.norm(dim=-1, keepdim=True) * cs
    elif view == 2:
        colors = colors.norm(dim=-1, keepdim=True) * ts

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, _ = rasterizer(
        means3D=m_cond,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=None,
        rotations=None,
        cov3D_precomp=cov)

    return rendered_image, cull_percent
