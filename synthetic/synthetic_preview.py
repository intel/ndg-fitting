# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: MIT License

import sys
import os

from torch_ema import ExponentialMovingAverage

sys.path += [".\\ext\\mitsuba3\\build\\Release\\python"]
os.environ["PATH"] += os.pathsep + ".\\ext\\mitsuba3\\build\\Release"

import mitsuba

# Set the desired mitsuba variant
mitsuba.set_variant('cuda_ad_rgb')

from synthetic.data_generation.variable_renderer import *
import configargparse
import imgui
import time
from imgui.integrations.glfw import GlfwRenderer
from utils import *
from generators.gaussian_generator import GaussianGenerator

exposure = [0.5]


def draw_sliders(renderer, custom_values):
    # Sensor sliders
    for i in range(len(renderer.variables_ids)):
        if renderer.variables[i] in renderer.sensors:
            var_id = renderer.variables_ids[i]
            if renderer.variables[i].num_parameters() == 1:
                changed, values = imgui.slider_float(var_id, *custom_values[var_id], 0, 1)
                values = [values]
            elif renderer.variables[i].num_parameters() == 2:
                changed, values = imgui.slider_float2(var_id, *custom_values[var_id], 0, 1)
            elif renderer.variables[i].num_parameters() == 3:
                changed, values = imgui.slider_float3(var_id, *custom_values[var_id], 0, 1)
            elif renderer.variables[i].num_parameters() == 4:
                changed, values = imgui.slider_float4(var_id, *custom_values[var_id], 0, 1)
            elif renderer.variables[i].num_parameters() == 5:
                changed, values = imgui.slider_float3(var_id + '_origin', *custom_values[var_id][0:3], 0, 1)
                changed2, values2 = imgui.slider_float2(var_id + '_target', *custom_values[var_id][3:5], 0, 1)
                values = values + values2
            elif renderer.variables[i].num_parameters() == 6:
                changed, values = imgui.slider_float3(var_id + '_origin', *custom_values[var_id][0:3], 0, 1)
                changed2, values2 = imgui.slider_float3(var_id + '_target', *custom_values[var_id][3:6], 0, 1)
                values = values + values2

            custom_values[var_id] = list(values)

        # Emitters sliders
    for i in range(len(renderer.variables_ids)):
        if renderer.variables[i] in renderer.emitters:
            var_id = renderer.variables_ids[i]
            if renderer.variables[i].num_parameters() == 1:
                changed, values = imgui.slider_float(var_id, *custom_values[var_id], 0, 1)
                values = [values]
            elif renderer.variables[i].num_parameters() == 2:
                changed, values = imgui.slider_float2(var_id, *custom_values[var_id], 0, 1)
            elif renderer.variables[i].num_parameters() == 3:
                changed, values = imgui.slider_float3(var_id, *custom_values[var_id], 0, 1)
            elif renderer.variables[i].num_parameters() == 4:
                changed, values = imgui.slider_float4(var_id, *custom_values[var_id], 0, 1)

            custom_values[var_id] = list(values)

        # Shapes sliders
    for i in range(len(renderer.variables_ids)):
        if renderer.variables[i] in renderer.shapes:
            var_id = renderer.variables_ids[i]
            if renderer.variables[i].num_parameters() == 1:
                changed, values = imgui.slider_float(var_id, *custom_values[var_id], 0, 1)
                values = [values]
            elif renderer.variables[i].num_parameters() == 2:
                changed, values = imgui.slider_float2(var_id, *custom_values[var_id], 0, 1)
            elif renderer.variables[i].num_parameters() == 3:
                changed, values = imgui.slider_float3(var_id, *custom_values[var_id], 0, 1)
            elif renderer.variables[i].num_parameters() == 4:
                changed, values = imgui.slider_float4(var_id, *custom_values[var_id], 0, 1)

            custom_values[var_id] = list(values)

        # Shape groups sliders
    for i in range(len(renderer.variables_ids)):
        if renderer.variables[i] in renderer.shapegroups:
            var_id = renderer.variables_ids[i]
            if renderer.variables[i].num_parameters() == 1:
                changed, values = imgui.slider_float(var_id, *custom_values[var_id], 0, 1)
                values = [values]
            elif renderer.variables[i].num_parameters() == 2:
                changed, values = imgui.slider_float2(var_id, *custom_values[var_id], 0, 1)

            custom_values[var_id] = list(values)

    # BSDFs sliders
    for i in range(len(renderer.variables_ids)):
        if renderer.variables[i] in renderer.bsdfs:
            var_id = renderer.variables_ids[i]
            if renderer.variables[i].num_parameters() == 1:
                changed, values = imgui.slider_float(var_id, *custom_values[var_id], 0, 1)
                values = [values]

            custom_values[var_id] = list(values)

    return custom_values


def get_samples(samples, num_samples):
    indices = random.choices(range(len(samples)), k=num_samples)

    inputs = []
    gt = []

    for batch in range(num_samples):
        inputs_batch, gt_batch = samples[indices[batch]]
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

    output = model(x.reshape(-1, c).to(device), view=view).reshape(n, h, w, -1)

    output = torch.where(emission > 1.0, emission, output + emission)

    return output


def run():
    preview_resolution = [800, 800]
    buffers_spp = 20
    gt_spp = 400

    conf = configargparse.ArgumentParser()

    # Directories
    conf.add('--scene_path', required=True, help='Path to the scene to preview')
    conf.add('--model', required=True, help='Model to load')

    # Misc
    conf.add('--seed', type=int, default=0, help='Seed for random numbers generator')
    conf.add('--device', type=str, default='cuda', help='Device to use for Pytorch training')

    conf = conf.parse_args()

    # Set random seeds
    np.random.seed(conf.seed)
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)

    renderer = VariableRenderer()

    renderer.load_scene(conf.scene_path)

    # Load pretrained GMM
    load_dict = torch.load(conf.model, map_location=conf.device)
    n_gs = load_dict["n_gs"]
    model = GaussianGenerator(dimensions=10+renderer.scene.total_parameters(), init_gs=torch.ones([n_gs, 10+renderer.scene.total_parameters()]),
                              ws_act=torch.exp, ws_act_inv=torch.log,
                              fs_act=torch.sigmoid, fs_act_inv=inverse_sigmoid,
                              ws_lr=0.1,
                              fs_lr=0.1,
                              m_sb_lr=0.01,
                              init_fs=torch.rand([n_gs, 3], device=conf.device),
                              n_projection_vectors=32)
    model.load_state_dict(load_dict["model"])
    model.ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    model.to(conf.device)

    # Initialize the random variables
    custom_values_preview = renderer.sample_values()

    # Initialize window
    window = impl_glfw_init()
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Initialize images
    prediction_img = np.zeros((preview_resolution[0], preview_resolution[1], 3))
    gt_img = np.zeros((preview_resolution[0], preview_resolution[1], 3))

    # Bind prediction texture
    prediction_id = bind_texture(prediction_img, preview_resolution)

    # Bind ground truth texture
    gt_id = bind_texture(gt_img, preview_resolution)

    # Enable key event callback
    glfw.set_key_callback(window, key_event)

    training_iterations = 0
    training_time = 0

    # Initialize options
    view = 0

    # Images are in linear space transform them to sRGB
    gl.glEnable(gl.GL_FRAMEBUFFER_SRGB)
    gl.glEnable(gl.GL_DITHER)

    start_frame = time.time()

    while not glfw.window_should_close(window):

        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        imgui.get_style().window_rounding = 0.0
        imgui.set_next_window_position(0, 0)
        imgui.begin('', imgui.WINDOW_NO_MOVE, imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)

        inputs, _ = renderer.get_custom_render(custom_values_preview, resolution=preview_resolution, spp=buffers_spp, need_gt=False)

        with (torch.no_grad(), model.ema.average_parameters()):
            prediction = evaluate(model, inputs, conf.device, inference=True, view=view)

        replace_texture(prediction, prediction_id, preview_resolution, exposure[0])

        if imgui.button('Generate GT'):
            _, gt = renderer.get_custom_render(custom_values_preview, resolution=preview_resolution, spp=gt_spp, need_gt=True)

            replace_texture(gt, gt_id, preview_resolution, exposure[0])

        imgui.image(prediction_id, preview_resolution[0], preview_resolution[1])
        imgui.same_line()
        imgui.image(gt_id, preview_resolution[0], preview_resolution[1])

        imgui.end()

        imgui.begin("Configurable parameters")

        custom_values_preview = draw_sliders(renderer, custom_values_preview)

        imgui.text('Number of Gaussians: '+str(model.n_gs + model.n_gs_sb))

        clicked, view = imgui.listbox(
            "View Type", view, ["Inference", "Partition"]
        )

        imgui.end()

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


def impl_glfw_init():
    width, height = 1920, 1080
    window_name = "Buffers Preview"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


def key_event(window, key, scancode, action, mods):
    if (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_P:
        exposure[0] = exposure[0] + 0.1
    elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_O:
        exposure[0] = exposure[0] - 0.1


if __name__ == "__main__":
    run()
