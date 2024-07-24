# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: MIT License

import time

import configargparse
import imgui
from imgui.integrations.glfw import GlfwRenderer

from generators.gaussian_generator import GaussianGenerator
from losses import *
from splatting.scene import Scene
from render import render
from utils import *
from simple_knn._C import distCUDA2


def sample_gs(scene, xyz_pcd, device):
    # XYZ
    xyz = xyz_pcd

    # View direction
    dir = torch.randn((xyz_pcd.shape[0], 3), device=device)
    dir = dir / dir.norm(dim=1, keepdim=True)

    params = torch.rand((xyz_pcd.shape[0], scene.total_parameters()), device=device)

    init_samples = torch.cat([xyz, dir, params], dim=-1)

    return init_samples


def run():
    preview_resolution = [1000, 800]

    conf = configargparse.ArgumentParser()

    # Directories
    conf.add('--dataset_path', required=True, help='Path to the dataset to be trained on')
    conf.add('--models_path', default='./models/splatting', help='Path to save the models')

    # Misc
    conf.add('--white_background', action='store_true', help='Whether to use white background')
    conf.add('--seed', type=int, default=0, help='Seed for random numbers generator')
    conf.add('--device', type=str, default='cuda', help='Device to use for Pytorch training')

    conf = conf.parse_args()

    # Set random seeds
    np.random.seed(conf.seed)
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)

    criterion_train = DssimL1Loss(permute=False)

    # Initialize window
    window = impl_glfw_init(width=1920, height=1080, window_name='Preview')
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Initialize images
    prediction_img = np.zeros((preview_resolution[0], preview_resolution[1], 3))
    gt_img = np.zeros((preview_resolution[0], preview_resolution[1], 3))

    # Bind prediction texture
    prediction_id = bind_texture(prediction_img, preview_resolution)

    # Bind ground truth texture
    gt_id = bind_texture(gt_img, preview_resolution)

    scene = Scene(conf.dataset_path, white_background=conf.white_background)

    xyz_pcd = torch.from_numpy(np.asarray(scene.data.point_cloud.points)).float().to(conf.device)
    rgb_pcd = torch.from_numpy(np.asarray(scene.data.point_cloud.colors)).float().to(conf.device)

    # Randomize colors if they are all zeros
    if rgb_pcd.max() == 0.0:
        rgb_pcd = torch.rand_like(rgb_pcd)

    # Compute scales in xyz
    dist2 = torch.clamp_min(distCUDA2(xyz_pcd), 1e-7)
    scales = (torch.sqrt(dist2))[..., None].repeat(1, 3)

    init_gs = sample_gs(scene, xyz_pcd=xyz_pcd, device=conf.device)

    cov_bias = torch.cat([scales, torch.tensor([[1.0, 1.0, 1.0]], device=conf.device).repeat(scales.shape[0], 1)], dim=-1)

    model = GaussianGenerator(dimensions=init_gs.shape[-1], init_gs=init_gs,
                              ws_act=lambda x: sigmoid(x), ws_act_inv=lambda x: inverse_sigmoid(x),
                              fs_act=lambda x: x, fs_act_inv=lambda x: x,
                              ws_lr=0.05,
                              fs_lr=0.005,
                              m_sb_lr=0.025,
                              init_ws=1e-1,
                              init_fs=rgb_pcd,
                              n_projection_vectors=8,
                              cov_bias=cov_bias)

    train_cameras = scene.train_cameras.copy()
    test_cameras = scene.test_cameras.copy()

    bg_color = [1, 1, 1] if conf.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Create model directory
    create_dir(os.path.join(conf.models_path, time.strftime('%Y%m%d-%H%M%S')))
    models_path = os.path.join(conf.models_path, time.strftime('%Y%m%d-%H%M%S'))

    training_iterations = 1

    train = False

    preview_camera_id = 0
    preview_view = 0

    losses_log = [0]

    while not glfw.window_should_close(window):
        # Training
        if train:
            model.optimizer.zero_grad()

            # Sample cameras
            train_camera_id = random.randint(0, len(train_cameras)-1)
            train_camera = train_cameras[train_camera_id]

            # Render using splatting
            render_result, _ = render(train_camera, model, background)

            prediction = render_result.unsqueeze(0)
            gt = train_camera.original_image.unsqueeze(0)

            # Loss
            loss = criterion_train(prediction, gt)
            loss.backward()

            model.optimizer.step()
            model.ema.update()

            losses_log.append(loss.item())

            if len(losses_log) > 200:
                losses_log.pop(0)

            training_iterations += 1

            if training_iterations == 300:
                model.prune_gs(0.1)
                model.seed_gs(0.1)
            elif training_iterations % 300 == 0:
                model.grow_gs_sb(0.1)
                model.seed_gs(0.1)

        # Previewing
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        imgui.get_style().window_rounding = 0.0
        imgui.set_next_window_position(0, 0)
        imgui.begin('Rendering', imgui.WINDOW_NO_MOVE, imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)

        imgui.text('Training iteration '+str(training_iterations))

        _clicked, train = imgui.checkbox('Training', train)

        clicked, preview_view = imgui.listbox(
            "View Type", preview_view, ["Inference", "Partition", "Type"]
        )

        changed, preview_camera_id = imgui.slider_int('Preview Camera', preview_camera_id, 0, len(test_cameras)-1)

        with (torch.no_grad(), model.ema.average_parameters()):
            preview_camera = test_cameras[preview_camera_id]

            # Render using splatting
            render_result, _ = render(preview_camera, model, background, view=preview_view)

            prediction = render_result.permute(1, 2, 0).unsqueeze(0)
            gt = preview_camera.original_image.permute(1, 2, 0).unsqueeze(0)

            replace_texture(prediction, prediction_id, preview_resolution)
            replace_texture(gt, gt_id, preview_resolution)

        imgui.image(prediction_id, preview_resolution[0], preview_resolution[1])
        imgui.same_line()
        imgui.image(gt_id, preview_resolution[0], preview_resolution[1])

        imgui.end()

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)


if __name__ == "__main__":
    run()