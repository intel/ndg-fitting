from matplotlib.transforms import offset_copy
import matplotlib.pyplot as plt
import numpy as np

import mitsuba.scalar_rgb as mi

def create_images(scene, params, parameter_changed, mult=1.0, increment=0.0):
    images = []
    for j in range(11):
        params['Material.' + parameter_changed] = j / 10 * mult + increment
        params.update()
        sensorIndex = 0
        mi.render(scene, seed=1, sensor_index=sensorIndex)
        # Get the scene's sensor (if many, can pick one by specifying the
        # index)
        sensor = scene.sensors()[sensorIndex]
        # The rendered data is stored in the film
        film = sensor.film()
        img = film.bitmap()
        img = img.convert(
            mi.Bitmap.PixelFormat.RGB,
            mi.Struct.Type.UInt8,
            srgb_gamma=True)
        image_np = np.array(img, copy=False)
        images.append(image_np)
    images.append((np.abs(images[10].astype("int") - images[0].astype("int"))))
    return images


# 1st scene for thin
scene = mi.load_file('scene_thin.xml')
params = mi.traverse(scene)

# spec_trans
params['Material.base_color.value'] = np.array([120, 20, 20]) / 256
params['Material.roughness.value'] = 0.01
params['Material.eta.value'] = 1.33
images_strans = create_images(scene, params, "spec_trans.value")


# roughness
params['Material.base_color.value'] = np.array([46, 47, 99]) / 256
params['Material.spec_trans.value'] = 1.0
images_roughness = create_images(scene, params, "roughness.value")

# eta
params['Material.base_color.value'] = np.array([15, 120, 12]) / 256
params['Material.roughness.value'] = 0.2
images_eta = create_images(scene, params, "eta.value", increment=1.001)

# Anisotropic
params['Material.base_color.value'] = np.array([93, 16, 55]) / 256
params['Material.roughness.value'] = 0.2
params['Material.eta.value'] = 1.33
images_anisotropic = create_images(scene, params, "anisotropic.value")

# 2nd scene for thin
scene = mi.load_file('scene_thin_light.xml')
params = mi.traverse(scene)
params['Material.base_color.value'] = np.array([150, 220, 150]) / 256
params['Material.roughness.value'] = 0.3
params['Material.anisotropic.value'] = 0.0
params['Material.spec_trans.value'] = 0.4
images_diff_trans = create_images(scene, params, "diff_trans.value", mult=2.0)

images_thin = []
images_thin.append(images_strans)
images_thin.append(images_roughness)
images_thin.append(images_eta)
images_thin.append(images_anisotropic)
images_thin.append(images_diff_trans)

pad = 1
parameters = [
    "spec_trans",
    "roughness",
    "eta(1.0-2.0)",
    "anisotropic",
    "diff_trans(0.0-2.0)"]
fig, axs = plt.subplots(len(parameters), 12, figsize=(12, 5), dpi=300)

for i in range(11):
    axs[0, i].set_title(str(i / 10), fontweight='semibold')
axs[0, 11].set_title("Diff.", fontweight='bold', color="red")

for j in range(len(parameters)):
    axs[j,
        0].annotate(parameters[j],
                    xy=(0,
                    5),
                    xytext=(-axs[j,
                                 0].yaxis.labelpad - pad,
                            0),
                    xycoords=axs[j,
                                 0].yaxis.label,
                    textcoords='offset points',
                    size='small',
                    ha='right',
                    va='center',
                    fontweight='semibold')

for i in range(12):
    for j in range(len(parameters)):
        axs[j, i].imshow(images_thin[j][i])
        axs[j, i].set_yticks([])
        axs[j, i].set_xticks([])
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("thinprincipled_blend.png", bbox_inches='tight', pad_inches=0.01)
