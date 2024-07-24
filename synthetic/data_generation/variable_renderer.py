# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: MIT License

import random

import tqdm

from utils import *

import mitsuba as mi


class VariableRenderer:

    def __init__(self, seed=0):

        self.seed = seed
        self.scene = None
        self.integrator = None
        self.params = None
        self.variables_ids = None
        self.variables = None
        self.sensors = None
        self.emitters = None
        self.shapes = None
        self.shapegroups = None
        self.bsdfs = None
        self.initial_values = None

    def load_scene(self, scene_filename):
        # Load the scene
        scene = mi.load_file(scene_filename)

        # Get the scene parameters
        params = mi.traverse(scene)

        # Scene randomizable objects
        emitters = scene.emitters()
        shapes = scene.shapes()
        bsdfs = scene.bsdfs()
        sensors = scene.sensors()
        shapegroups = scene.shapegroups()

        # Variables parameters
        variables = []
        variables_ids = []

        # Retrieve chosen variable objects
        for i in range(len(shapes)):
            if 'var' in shapes[i].id():
                variables_ids.append(shapes[i].id())
                variables.append(shapes[i])

        for i in range(len(shapegroups)):
            if 'var' in shapegroups[i].id():
                variables_ids.append(shapegroups[i].id())
                variables.append(shapegroups[i])

        for i in range(len(emitters)):
            if 'var' in emitters[i].id():
                variables_ids.append(emitters[i].id())
                variables.append(emitters[i])

        for i in range(len(bsdfs)):
            if 'var' in bsdfs[i].id():
                variables_ids.append(bsdfs[i].id())
                variables.append(bsdfs[i])

        for i in range(len(sensors)):
            if 'var' in sensors[i].id():
                variables_ids.append(sensors[i].id())
                variables.append(sensors[i])

        self.scene = scene
        self.integrator = scene.integrator()
        self.params = params
        self.variables_ids = variables_ids
        self.variables = variables
        self.sensors = sensors
        self.emitters = emitters
        self.shapes = shapes
        self.shapegroups = shapegroups
        self.bsdfs = bsdfs

    def setup_scene(self, custom_values):
        for i in range(len(self.variables)):

            # Emitters
            if self.variables[i] in self.emitters:
                assert self.variables[i].num_parameters() in [1, 3], "Emitters need 1 or 3 parameters, defined in the xml as num_parameters"

                if self.variables[i].is_environment():
                    # TODO: WIP implementation of rotating environment map, currently it is rotated in a fixed range
                    rotation_axis = np.array([0.0, 1.0, 0.0])

                    rel_angle = custom_values[self.variables[i].id()][0]
                    angle = 100 + rel_angle*100

                    rotation = rotation_matrix(rotation_axis, angle)
                    translation = translation_matrix(np.array([0.0, 0.0, 0.0]))

                    self.params[self.variables[i].id() + '.to_world'] = (translation @ rotation)
                else:
                    radiance = self.variables[i].min_bounds() + custom_values[self.variables[i].id()][0] * self.variables[i].range_bounds()

                    self.params[self.variables[i].id() + '.radiance.value'] = radiance
            # Sensors
            elif self.variables[i] in self.sensors:
                assert self.variables[i].num_parameters() in [5], "Sensors need 5 parameters, defined in the xml as num_parameters"

                bbox_range = self.scene.bbox().extents()

                origin = np.array(self.variables[i].min_bounds() + custom_values[self.variables[i].id()][0:3] * self.variables[i].range_bounds())[0]

                if self.variables[i].num_parameters() == 5:
                    # Target ranges based on bbox
                    target = np.array([self.scene.bbox().min[0] + bbox_range.x / 3 + (custom_values[self.variables[i].id()][3] * bbox_range.x / 3),
                                       self.scene.bbox().min[1] + (bbox_range.y / 3),
                                       self.scene.bbox().min[2] + bbox_range.z / 3 + (custom_values[self.variables[i].id()][4] * bbox_range.z / 3)])

                self.params[self.variables[i].id() + '.to_world'] = camera_to_world(origin, target, np.array([0, 1, 0]))

            # Shapes
            elif self.variables[i] in self.shapes:
                assert self.variables[i].num_parameters() in [1, 2, 3], "Shapes need 1, 2 or 3 parameters, defined in the xml as num_parameters"

                param_index = 0

                custom_vector = [0, 0, 0]

                # If x has a range use it as a parameter
                if self.variables[i].range_bounds().x[0] > 0:
                    custom_vector[0] = custom_values[self.variables[i].id()][param_index]
                    param_index += 1

                # If y has a range use it as a parameter
                if self.variables[i].range_bounds().y[0] > 0:
                    custom_vector[1] = custom_values[self.variables[i].id()][param_index]
                    param_index += 1

                # If z has a range use it as a parameter
                if self.variables[i].range_bounds().z[0] > 0:
                    custom_vector[2] = custom_values[self.variables[i].id()][param_index]
                    param_index += 1

                translation = translation_matrix(np.array(self.variables[i].min_bounds() + custom_vector * self.variables[i].range_bounds()))
                rotation = np.eye(4)

                # The parameter left is for rotation
                if self.variables[i].num_parameters() > param_index:
                    rotation_axis = np.array(self.variables[i].rotation_axis())[0]
                    angle = (self.variables[i].min_angle() + custom_values[self.variables[i].id()][param_index] * self.variables[i].range_angle())[0]

                    rotation = rotation_matrix(rotation_axis, angle)

                self.params[self.variables[i].id() + '.to_world'] = (translation @ rotation)

            # BSDFs
            elif self.variables[i] in self.bsdfs:
                # TODO: add BSDF variability
                pass

        self.params.update()

    def sample_values(self):
        custom_values = dict()

        for j in range(len(self.variables_ids)):
            var_id = self.variables_ids[j]
            parameters = []
            for k in range(self.variables[j].num_parameters()):
                parameters.append(random.uniform(0, 1))

            custom_values[var_id] = parameters

        return custom_values

    def sample_inputs(self, resolution, spp, samples_per_instance, num_instances):
        input_samples = []

        for i in tqdm.tqdm(range(num_instances)):
            custom_values_train = self.sample_values()

            inputs, _ = self.get_custom_render(custom_values_train, stochastic=True, resolution=resolution, spp=spp, need_gt=False)

            inputs = inputs.reshape(-1, inputs.shape[-1])[:, 3:]

            ids = np.random.choice(inputs.shape[0], size=samples_per_instance)

            samples = inputs[ids]

            input_samples.append(samples)

        input_samples = torch.cat(input_samples, dim=0)

        return input_samples

    def get_custom_render(self, custom_values, resolution, spp, stochastic=False, need_gt=True):

        # Set up the scene for the given custom  values and check intersection
        self.setup_scene(custom_values)

        # Whether to render the gt image
        self.integrator.set_nested(need_gt)

        # Set target resolution
        self.sensors[0].film().set_size(resolution)

        # Choose seed based on whether a stochastic rendering is requested
        seed = 0
        if stochastic:
            seed = self.seed
            self.seed += 1

        # Call the scene's integrator to render the loaded scene
        result = self.integrator.render(self.scene, self.sensors[0], seed=seed, spp=spp).torch()

        gt = []

        if need_gt:
            gt = result[:, :, 0:3]
            gt = gt.unsqueeze(0)

        emission = result[:, :, 3:6]

        # Normalize positions from bounding box
        positions = result[:, :, 6:9]
        bbox_range = self.scene.bbox().max - self.scene.bbox().min
        positions = (positions - self.scene.bbox().min.torch().to('cuda')) / bbox_range.torch().to('cuda')
        positions = positions*2.0-1.0

        wi = result[:, :, 9:12]
        albedo = result[:, :, 12:15]
        alpha = result[:, :, 15:16]

        buffers = [emission, positions, wi, albedo, alpha]

        inputs = stack_inputs_tensor(buffers, self.variables, [*custom_values.values()])
        inputs = inputs.unsqueeze(0)

        return inputs, gt

    def get_camera_matrix(self, custom_values):
        for i in range(len(self.variables)):

            camera_matrix = None

            # Sensors
            if self.variables[i] in self.sensors:
                assert self.variables[i].num_parameters() in [5], "Sensors need 5 parameters, defined in the xml as num_parameters"

                bbox_range = self.scene.bbox().extents()

                origin = np.array(self.variables[i].min_bounds() + custom_values[self.variables[i].id()][0:3] * self.variables[i].range_bounds())[0]

                if self.variables[i].num_parameters() == 5:
                    # Target ranges based on bbox
                    target = np.array([self.scene.bbox().min[0] + bbox_range.x / 3 + (custom_values[self.variables[i].id()][3] * bbox_range.x / 3),
                                       self.scene.bbox().min[1] + (bbox_range.y / 3),
                                       self.scene.bbox().min[2] + bbox_range.z / 3 + (custom_values[self.variables[i].id()][4] * bbox_range.z / 3)])

                camera_matrix = camera_to_world(origin, target, np.array([0, 1, 0]))

        return camera_matrix

