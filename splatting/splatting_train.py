# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: MIT License

import copy
import time

import configargparse
import tqdm
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from simple_knn._C import distCUDA2

from generators.gaussian_generator import GaussianGenerator
from losses import *
from splatting.scene import Scene
from render import render
from utils import *


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

    conf = configargparse.ArgumentParser()

    # Directories
    conf.add('--dataset_path', required=True, help='Path to the dataset to be trained on')
    conf.add('--models_path', default='./models/splatting', help='Path to save the models')

    # Misc
    conf.add('--white_background', action='store_true', help='Whether to use white background')
    conf.add('--seed', type=int, default=0, help='Seed for random numbers generator')
    conf.add('--device', type=str, default='cuda', help='Device to use for Pytorch training')
    conf.add('--tensorboard', action='store_true', help='Whether to use tensorboard for visualization')

    conf = conf.parse_args()

    # Set random seeds
    np.random.seed(conf.seed)
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)

    if conf.tensorboard:
        summary_writer = SummaryWriter('./runs/splatting/'+time.strftime('%Y%m%d-%H%M%S'))

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    criterion_train = DssimL1Loss(permute=False)
    criterion_test = AllMetrics()

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

    test = True

    losses_log = [0]

    for k in tqdm(range(30000), desc='Training model'):
        iter_start.record()

        model.optimizer.zero_grad()

        # Sample cameras
        train_camera_id = random.randint(0, len(train_cameras)-1)
        train_camera = train_cameras[train_camera_id]

        # Render using splatting
        render_result, cull_percent = render(train_camera, model, background)

        prediction = render_result.unsqueeze(0)
        gt = train_camera.original_image.unsqueeze(0)

        # Loss
        loss = criterion_train(prediction, gt)
        loss.backward()

        model.optimizer.step()
        model.ema.update()

        losses_log.append(loss.item())

        iter_end.record()

        iter_start.synchronize()
        iter_end.synchronize()

        if conf.tensorboard:
            summary_writer.add_scalar('Iteration time', iter_start.elapsed_time(iter_end), training_iterations)
            summary_writer.add_scalar('Cull percent', cull_percent, training_iterations)

        if len(losses_log) > 200:
            losses_log.pop(0)

        training_iterations += 1

        if training_iterations % 1000 == 0:
            model_to_save = copy.deepcopy(model)
            model_to_save.finalize_gs()

            save_dict = {
                'training_iterations': training_iterations,
                'n_gs': model_to_save.n_gs,
                'model': model_to_save.state_dict()
            }

            torch.save(save_dict, models_path + '/model' + str(training_iterations) + '.pth')
            model_to_save.save_ply(models_path + '/point_cloud' + str(training_iterations) + '.ply')

        if test and training_iterations % 1000 == 0:
            with (torch.no_grad(), model.ema.average_parameters()):
                criterion_test.reset()

                test_imgs = []

                for i, test_camera in enumerate(test_cameras):
                    render_result, _ = render(test_camera, model, background)

                    prediction = render_result.unsqueeze(0)
                    gt = test_camera.original_image.unsqueeze(0)

                    criterion_test(prediction.permute(0, 2, 3, 1), gt.permute(0, 2, 3, 1))

                    test_imgs.append(prediction.cpu())
                    test_imgs.append(gt.cpu())

                test_imgs = torch.cat(test_imgs, dim=0)

                if conf.tensorboard:
                    summary_writer.add_scalar('Number of Gaussians', model.n_gs + model.n_gs_sb, training_iterations)

                    summary_writer.add_scalar('Test Loss', criterion_test.metrics['psnr'] / criterion_test.samples, training_iterations)

                    summary_writer.add_images('Test Samples', test_imgs, global_step=training_iterations)

        if training_iterations == 300:
            model.prune_gs(0.1)
            model.seed_gs(0.1)
        elif training_iterations % 300 == 0:
            model.grow_gs_sb(0.1)
            model.seed_gs(0.1)


if __name__ == "__main__":
    run()