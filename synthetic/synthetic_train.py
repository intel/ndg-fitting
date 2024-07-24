# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: MIT License

import sys
import os

sys.path += [".\\ext\\mitsuba3\\build\\Release\\python"]
os.environ["PATH"] += os.pathsep + ".\\ext\\mitsuba3\\build\\Release"

import mitsuba

# Set the desired mitsuba variant
mitsuba.set_variant('cuda_ad_rgb')

import copy
import json
from torch.utils.tensorboard import SummaryWriter
from synthetic.data_generation.variable_renderer import *
import configargparse
import time
from utils import *
from losses import *
from generators.gaussian_generator import GaussianGenerator


def get_samples(samples, num_samples):
    indices = random.choices(range(len(samples)), k=num_samples)

    inputs = []
    gt = []

    for batch in range(num_samples):
        inputs_batch, gt_batch = samples[indices[batch]]

        # Ensure no nans in ground truths
        gt_batch = torch.nan_to_num(gt_batch, nan=0)

        inputs.append(inputs_batch)
        gt.append(gt_batch)

    inputs = torch.cat(inputs, dim=0)
    gt = torch.cat(gt, dim=0)

    return inputs, gt


def evaluate(model, inputs, device, inference=False, view=0):
    # Get emission and position from input
    emission = inputs[:, :, :, 0:3]

    # Ignore emission
    x = inputs[:, :, :, 3:]

    n, h, w, c = x.shape

    output = model(x.reshape(-1, c), inference=inference, view=view).reshape(n, h, w, -1)

    output = torch.where(emission > 1.0, emission, output + emission)

    return output


def run():
    training_resolution = [128, 128]

    conf = configargparse.ArgumentParser()

    # Directories
    conf.add('--scene_path', required=True, help='Path to the scene to be trained on')
    conf.add('--dataset_path', required=True, help='Path to the dataset to be trained on')
    conf.add('--models_path', default='./models/synthetic', help='Path to save the models')

    # Misc
    conf.add('--batch_size', type=int, default=8, help='Batch size')
    conf.add('--tensorboard', action='store_true', help='Whether to use tensorboard for visualization')
    conf.add('--seed', type=int, default=0, help='Seed for random numbers generator')
    conf.add('--device', type=str, default='cuda', help='Device to use for Pytorch training')

    conf = conf.parse_args()

    # Set random seeds
    np.random.seed(conf.seed)
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)

    if conf.tensorboard:
        summary_writer = SummaryWriter('./runs/synthetic/'+time.strftime('%Y%m%d-%H%M%S'))

    renderer = VariableRenderer()

    renderer.load_scene(conf.scene_path)

    criterion_train = RelL2Loss()
    criterion_test = AllMetrics()

    init_gs = renderer.sample_inputs(training_resolution, spp=40, samples_per_instance=32, num_instances=32)
    init_fs = torch.rand([init_gs.shape[0], 3], device=init_gs.device)
    model = GaussianGenerator(dimensions=10+renderer.scene.total_parameters(), init_gs=init_gs,
                              ws_act=torch.exp, ws_act_inv=torch.log,
                              fs_act=torch.sigmoid, fs_act_inv=inverse_sigmoid,
                              ws_lr=0.1,
                              fs_lr=0.1,
                              m_sb_lr=0.015,
                              init_fs=init_fs,
                              n_projection_vectors=32,
                              cov_bias=1e-1,
                              betas=(0.9, 0.99))

    batch_size = conf.batch_size

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    train_samples = []
    test_samples = []

    # Create model directory
    create_dir(os.path.join(conf.models_path, time.strftime('%Y%m%d-%H%M%S')))
    models_path = os.path.join(conf.models_path, time.strftime('%Y%m%d-%H%M%S'))

    # Read dataset
    path_transforms = os.path.join(conf.dataset_path, 'transforms_train.json')
    with open(path_transforms) as pose_file:
        transforms = json.load(pose_file)

    for idx, out in tqdm.tqdm(enumerate(transforms['frames'])):
        sample = np.load(os.path.join(conf.dataset_path, out['file_path'])+'.npz')

        if idx % 8 != 0:
            train_samples.append((torch.from_numpy(sample['inputs']).unsqueeze(0), torch.from_numpy(sample['gt']).unsqueeze(0)))
        else:
            test_samples.append((torch.from_numpy(sample['inputs']).unsqueeze(0), torch.from_numpy(sample['gt']).unsqueeze(0)))

    training_iterations = 0

    for k in tqdm.tqdm(range(30000), desc='Training model'):

        inputs, gt = get_samples(train_samples, batch_size)
        inputs = torch.nn.functional.interpolate(inputs.permute(0, 3, 1, 2), size=training_resolution, mode='bilinear').permute(0, 2, 3, 1)
        gt = torch.nn.functional.interpolate(gt.permute(0, 3, 1, 2), size=training_resolution, mode='bilinear').permute(0, 2, 3, 1)

        iter_start.record()

        model.optimizer.zero_grad()

        outputs = evaluate(model, inputs.to(conf.device), conf.device)

        loss = criterion_train(outputs, gt.to(conf.device))
        loss.backward()

        model.optimizer.step()
        model.ema.update()
        model.update_lr(training_iterations)

        iter_end.record()

        iter_start.synchronize()
        iter_end.synchronize()

        if conf.tensorboard:
            summary_writer.add_scalar('Iteration time', iter_start.elapsed_time(iter_end), training_iterations)

        training_iterations += 1

        if training_iterations % 1000 == 0:
            model_to_save = copy.deepcopy(model)
            model_to_save.finalize_gs()
            model_to_save.prune_gs(0.01)

            save_dict = {
                'training_iterations': training_iterations,
                'n_gs': model_to_save.n_gs,
                'model': model_to_save.state_dict()
            }

            torch.save(save_dict, models_path + '/model' + str(training_iterations) + '.pth')

        if training_iterations % 1000 == 0:
            with (torch.no_grad(), model.ema.average_parameters()):
                criterion_test.reset()

                test_imgs = []

                for test_inputs, test_gt in test_samples:
                    test_output = evaluate(model, test_inputs.to(conf.device), conf.device)

                    criterion_test(test_output, test_gt.to(conf.device))

                    test_imgs.append(linear_to_srgb(test_output.cpu()))
                    test_imgs.append(linear_to_srgb(test_gt.cpu()))

                test_imgs = torch.cat(test_imgs, dim=0)

                if conf.tensorboard:
                    summary_writer.add_scalar('Number of Gaussians', model.n_gs + model.n_gs_sb, training_iterations)

                    summary_writer.add_scalar('Test DSSIM Loss', criterion_test.metrics['dssim'] / criterion_test.samples, training_iterations)
                    summary_writer.add_scalar('Test L1 Loss', criterion_test.metrics['l1'] / criterion_test.samples, training_iterations)

                    summary_writer.add_images('Test Samples', test_imgs.permute(0, 3, 1, 2), global_step=training_iterations)

        if training_iterations == 300:
            model.prune_gs(0.1)
            model.seed_gs(0.1)
        elif training_iterations % 300 == 0:
            model.grow_gs_sb(0.1)
            model.seed_gs(0.1)

        if training_iterations % 2000 == 0 and training_resolution[0] < 256:
            training_resolution[0] *= 2
            training_resolution[0] = min(training_resolution[0], 256)
            training_resolution[1] *= 2
            training_resolution[1] = min(training_resolution[1], 256)

            batch_size //= 2
            batch_size = max(batch_size, 1)


if __name__ == "__main__":
    run()
