"""
Usage: denoiser_ref.py {optixdenoiser_bin}

This script is used to generate the reference images for the OptiX denoiser
tests.

The first and only argument of this script `optixdenoiser_bin` is the path to
the `optixDenoiser` executable which is built from the OptiX 7.4 SDK examples.
"""

import mitsuba as mi
import drjit as dr
import sys
import subprocess

mi.set_variant('cuda_ad_rgb')

scene = mi.load_file("../../scenes/cbox/cbox-rgb.xml", res=64)
sensor = scene.sensors()[0]
integrator = mi.load_dict({
    'type': 'aov',
    'aovs': 'albedo:albedo,sh_normal:sh_normal',
    'img': {
        'type': 'path',
        'max_depth' : 6,
    }
})

mi.render(scene, spp=2, integrator=integrator, sensor=sensor)
mutlichannel = sensor.film().bitmap()
aovs = dict(mutlichannel.split())

# Transform normals into sensor frame
normals = mi.TensorXf(aovs['sh_normal'])
new_normals = dr.zeros(mi.Normal3f, normals.shape[0] * normals.shape[1])
for i in range(3):
    new_normals[i] = dr.ravel(normals)[i::3]
t = sensor.world_transform()
new_normals = t.inverse() @ new_normals
new_normals[0] = -new_normals[0]
new_normals[2] = -new_normals[2]
for i in range(3):
    normals[..., i] = new_normals[i]
aovs['sh_normals'] = mi.Bitmap(normals)

flow_shape = list(mi.TensorXf(aovs['<root>']).shape)
flow_shape[2] = 2
flow = mi.Bitmap(dr.zeros(mi.TensorXf, shape=flow_shape))

aovs['<root>'].write('noisy.exr')
aovs['albedo'].write('albedo.exr')
aovs['sh_normals'].write('normals.exr')
flow.write('flow.exr')

optixDenoiser_bin = sys.argv[1]
print(optixDenoiser_bin)

process = subprocess.run([optixDenoiser_bin, "-o", "ref.exr", "noisy.exr"], check=True)
process = subprocess.run([optixDenoiser_bin, "-a", "albedo.exr", "-o", "ref_albedo.exr", "noisy.exr"], check=True)
process = subprocess.run([optixDenoiser_bin, "-a", "albedo.exr", "-o", "ref_albedo.exr", "noisy.exr"], check=True)
process = subprocess.run([optixDenoiser_bin, "-a", "albedo.exr", "-n", "normals.exr", "-o", "ref_normals.exr", "noisy.exr"], check=True)
process = subprocess.run([optixDenoiser_bin, "-a", "albedo.exr", "-n", "normals.exr", "-F", "1-1", "-f", "flow.exr", "-o", "ref_temporal.exr", "noisy.exr"], check=True)
