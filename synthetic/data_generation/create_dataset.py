# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: MIT License

import sys
import os

sys.path += [".\\ext\\mitsuba3\\build\\Release\\python"]
os.environ["PATH"] += os.pathsep + ".\\ext\\mitsuba3\\build\\Release"

import mitsuba

# Set the desired mitsuba variant
mitsuba.set_variant('cuda_ad_rgb')

import time
import json
from synthetic.data_generation.variable_renderer import *
import configargparse
from utils import *


conf = configargparse.ArgumentParser()

# Directories
conf.add('--scene_path', required=True, help='Path to the scene to be trained on')
conf.add('--datasets_path', default='./datasets/synthetic', help='Path to the dataset outuput path')

# Training parameters
conf.add('--n_samples', type=int, default=1000, help='Number of training samples')
conf.add('--n_passes', type=int, default=1, help='Number of training samples')

# Misc
conf.add('--seed', type=int, default=0, help='Seed for random numbers generator')
conf.add('--spp', type=int, default=4000, help='Samples per pixel')
conf.add('--device', type=str, default='cuda', help='Device to use for Pytorch training')

conf = conf.parse_args()

resolution = [256, 256]

# Set random seeds
np.random.seed(conf.seed)
random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)

renderer = VariableRenderer()
renderer.load_scene(conf.scene_path)

# Create a folder for the dataset and save a copy of the configs
dataset_dir = os.path.join(conf.datasets_path + '/' + time.strftime('%Y%m%d-%H%M%S'))
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(dataset_dir+'/train', exist_ok=True)

path_transforms = os.path.join(dataset_dir, 'transforms_train.json')

transforms = {}
transforms['frames'] = []

for i in tqdm.tqdm(range(conf.n_samples), desc="Rendering dataset"):

    filename = 'out_' + str(i)

    custom_values = renderer.sample_values()

    inputs, gt = renderer.get_custom_render(custom_values, stochastic=True, resolution=resolution, spp=conf.spp, need_gt=True)

    for j in range(conf.n_passes):
        inputs_pass, gt_pass = renderer.get_custom_render(custom_values, stochastic=True, resolution=resolution, spp=conf.spp, need_gt=True)

        inputs += inputs_pass
        gt += gt_pass

    if conf.n_passes > 0:
        inputs /= (conf.n_passes+1)
        gt /= (conf.n_passes+1)

    camera_matrix = matrix_to_list(renderer.get_camera_matrix(custom_values))
    params = get_params(renderer.variables, [*custom_values.values()])

    transforms['frames'].append({
        'file_path': './' + 'train' + '/' + filename,
        'transform_matrix': matrix_to_list(camera_matrix),
        'parameters': params
    })

    with open(path_transforms, 'w+') as pose_file:
        json.dump(transforms, pose_file, sort_keys=False, indent=4)

    np.savez(os.path.join(dataset_dir+'\\train\\'+'out_'+str(i)), gt=gt[0].cpu().numpy(), inputs=inputs[0].cpu().numpy())
