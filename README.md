# N-Dimensional Gaussians for Fitting of High Dimensional Functions

![Teaser](https://www.sdiolatz.info/ndg-fitting/static/images/teaser.png)

Stavros Diolatzis, Tobias Zirr, Alexandr Kuznetsov, Georgios Kopanas, Anton Kaplanyan

<img src="https://www.sdiolatz.info/ndg-fitting/static/images/intel_logo.png" alt="intel" width="100"/> <img src="https://www.sdiolatz.info/ndg-fitting/static/images/inria_logo.png" alt="inria" width="140"/> <img src="https://www.sdiolatz.info/ndg-fitting/static/images/uca_logo.png" alt="ucda" width="200"/>

### [Project Website](https://www.sdiolatz.info/ndg-fitting/)

Abstract: In the wake of many new ML-inspired approaches for reconstructing and representing high-quality 3D content, recent hybrid and explicitly learned representations exhibit promising performance and quality characteristics. However, their scaling to higher dimensions is challenging, e.g. when accounting for dynamic content with respect to additional parameters such as material properties, illumination, or time. In this paper, we tackle these challenges for an explicit representations based on Gaussian mixture models. With our solutions, we arrive at efficient fitting of compact N-dimensional Gaussian mixtures and enable efficient evaluation at render time: For fast fitting and evaluation, we introduce a high-dimensional culling scheme that efficiently bounds N-D Gaussians, inspired by Locality Sensitive Hashing. For adaptive refinement yet compact representation, we introduce a loss-adaptive density control scheme that incrementally guides the use of additional capacity towards missing details. With these tools we can for the first time represent complex appearance that depends on many input dimensions beyond position or viewing angle within a compact, explicit representation optimized in minutes and rendered in milliseconds.

## Code Notes

This codebase was developed and tested on Windows, we would like to support more platforms but for now we haven't tested this on them.

### Motivation

This code base is meant to be used to replace the usage of neural networks (mainly MLPs) with N-Dimensional Gaussian mixtures to improve training from hours to minutes. 

Ideal scenarios are cases where the target application has many input dimensions with strong inter-dependencies. We demonstrate two scenarios:

* Global Illumination with 10D+: MLPs have been used to turn G-Buffer renderings into global illumination images in previous work but usually took many hours. Our mixture can be trained in minutes.

* Radiance Fields with a focus on view-dependcy 6D: In this application we are inspired by 3D Gaussian Splatting but instead of considering only the world xyz we consider it together with the view direction xyz allowing the Gaussians to represent complex view dependent effects. In this case we don't evaluate the whole mixture but first project to 3D and then use splatting to render the final image.

## Cloning the repository

Use `recursive` when cloning the repo:

```
git clone --recursive https://github.com/intel/ndg-fitting.git
```

## Code Overview

The codebase includes application specific scripts and code that is common between applications. Here is a list of the main scripts and what they represent:

* `generators\gaussian_generator`: The implementation of the N-Dimensional Gaussian Generator which has the implementation of the optimizer controller main-sub Gaussian densification.
* `synthetic\`: Scripts related to the 10D+ Global Illumination application
* `splatting\`: Scripts related to the 6D Radiance Fields application

## 6D Radiance Fields `(./splatting)`

In this application we use 6D dimensional Gaussians parameterized by world position and view direction. For a given camera all Gaussians are projected to 3D and are then splatted using the rasterizer from https://github.com/graphdeco-inria/gaussian-splatting.

### Training 

First install all the requirements in `requirements.txt` including the `ext\diff-gaussian-rasterization` and the `ext\simple-knn` modules using pip.

To train first you should have a compatible scene. We currently support COLMAP and Blender type scenes (for information to turn videos into these formats follow information at https://github.com/graphdeco-inria/gaussian-splatting). Once you have your scene you can train offline with:

```
./splatting/splatting_train.py --dataset_path <path_to_scene> --tensorboard
```

This will start training and it will save in the .\models\ path a .pth and .ply file every 1000 iterations. It will also track progress using tensorboard like test PSNR, number of Gaussians etc.

### Training with Preview

![Preview](https://github.com/user-attachments/assets/d91ee07d-4310-46b5-a563-c6c463670aa3)

Alternative you can run a training with an online preview of the test images using ImGUI. This is slightly slower but allows you to inpsect training, observe how the refinement works etc.

```
./splatting/splatting_train.py --dataset_path <path_to_scene>
```

### Pretrained Models

The CD and Tools models shown in the paper are pending approval and should be up by late August - early September. Alternatively you can train on theses scenes by downloading them from here: https://nex-mpi.github.io/

### Note on 360 scenes

Currently our method and specifically the refinement does not support 360 scenes. In cases where the initialization of Gaussians is very approximate the refinement cannot overcome the local minima. We are aiming to adapt our refinement for that scenario in the future. For now you either need a good initialization or mostly front facing scenes.

## Global Illumination `(./synthetic)`

In this application we take a **custom** Mitsuba 3 scene with optional variability, render buffers and evaluate the Gaussian Mixture on the surfaces of the scene.

First you need to compile the custom Mistuba 3 version in `.ext/mistuba3` directory. For instructions on how to do this follow https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html. 

## Training

Convert a scene into one compatible with our method. This comes down to changing the `.xml` to include rendering the necessary buffers and to define the camera to be be variable for a given range:
```
    <integrator type="aov">
        <string name="aovs" value="1emission:emission, 2position:position, 3wi:wi, 4albedo:albedo, 5alpha:alpha, 6normals:sh_normal" />
        <integrator type="path" name="image">
			<integer name="max_depth" value="5"/>
            <boolean name="hide_emitters" value="false" />
        </integrator>
    </integrator>
    <sensor type="perspective" id="var_sensor">
        <float name="fov" value="60" />
        <transform name="to_world">
            <matrix value="-0.993341 -0.0130485 -0.114467 4.44315 0 0.993565 -0.11326 16.9344 0.115208 -0.112506 -0.98695 49.9102 0 0 0 1" />
        </transform>
		<sampler type="independent">
            <integer name="sample_count" value="40" />
        </sampler>
         <film type="hdrfilm">
            <integer name="width" value="256" />
            <integer name="height" value="256" />
            <rfilter type="tent" />
			<string name="component_format" value="float32"/>
        </film>
		<vector name="min_bounds" value="-10, 17, 20"/>
		<vector name="range_bounds" value="20, 2, 25"/>
		<integer name="num_parameters" value="5"/>
    </sensor>
```

Make sure to adjust min_bounds and range bounds for the range of possible camera placement.

Once you have this xml and the scene ready you can create a dataset to train on in the following way:

```
./synthetic/data_generation/create_dataset.py --scene_path <path_to_scene_xml>
```

Once you have the generated dataset you can train on the synethetic scene. To do this run:

```
./synthetic/synthetic_train.py -- scene_path <path_to_scene_xml> --dataset_path <path_to_dataset>
```

## Preview Trained Model

![Preview](https://github.com/user-attachments/assets/7a44192c-4789-4c35-9748-d8b7d33a8788)

To render using the trained model use the preview script:

```
./synthetic/synthetic_preview.py --scene_path <path_to_scene_xml> --model <path_to_model_pth>
```

## Adapting the method your own application

We believe that our N-Dimensional Gaussian Generator is a good fit for various applications. To use it in your applications here are a few tips:

* Use the GaussianGenerator as a model instead of an MLP
* Start with the learning rates used in either application
* Initialize the generator with a few Gaussians (you can samle your input space) and give a starting covariance bias (i.e the Gaussains are starting isotropic, how much should they span in each dimension)
* Usually it's good to normalzie your inputs (as you do for MLPs)
* Follow our applications to frequently prune, seed and grow the Gaussians
* If you want to evaluate the whole N-Dimensional mixture use its forward function
* Alternatively if you want to handle the Gaussians in some other way (like we project to 3D) use the `get_gs_total` function

We aim to improve this apect of the method, so your feedback is valuable.

## Future TODOs

- [ ] Upload pretrained models and scenes

