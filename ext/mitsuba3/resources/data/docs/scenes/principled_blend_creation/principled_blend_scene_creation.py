from matplotlib.transforms import offset_copy
import matplotlib.pyplot as plt
import numpy as np

import mitsuba.scalar_rgb as mi

def create_images(scene, params, parameter_changed):
    images = []
    for j in range(11):
        params['Material.' + parameter_changed] = j / 10
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


# 1st scene for regular
scene = mi.load_file('scene.xml')
params = mi.traverse(scene)

# roughness
params['Material.base_color.value'] = np.array([46 / 256, 47 / 256, 99 / 256])
params['Material.roughness.value'] = 0.01
params['Material.specular'] = 1
images_roughness = create_images(scene, params, "roughness.value")

# metallic
params['Material.base_color.value'] = np.array([220, 180, 31]) / 256
params['Material.roughness.value'] = 0.2
params['Material.specular'] = 0.8
images_metallic = create_images(scene, params, "metallic.value")

# Anisotropic
params['Material.base_color.value'] = np.array([93, 16, 55]) / 256
params['Material.roughness.value'] = 0.3
params['Material.anisotropic.value'] = 0.0
params['Material.metallic.value'] = 0.8
images_anisotropic = create_images(scene, params, "anisotropic.value")

# specular
params['Material.base_color.value'] = np.array([250, 10, 10]) / 256
params['Material.roughness.value'] = 0.01
params['Material.anisotropic.value'] = 0.0
params['Material.metallic.value'] = 0.0
params['Material.spec_tint.value'] = 0.0
params['Material.specular'] = 0.0
images_specular = create_images(scene, params, "specular")
images_specular_tint = create_images(scene, params, "spec_tint.value")

# flatness
params['Material.base_color.value'] = np.array([130, 190, 130]) / 256
params['Material.roughness.value'] = 0.4
params['Material.spec_tint.value'] = 0.0
params['Material.flatness.value'] = 0.0
params['Material.specular'] = 0
images_flatness = create_images(scene, params, "flatness.value")

# spec_trans
params['Material.base_color.value'] = np.array([256, 256, 256]) / 256
params['Material.roughness.value'] = 0.01
params['Material.specular'] = 0.5
images_strans = create_images(scene, params, "spec_trans.value")

# metallic spec_trans=1
params['Material.base_color.value'] = np.array([220, 180, 31]) / 256
images_st_metallic = create_images(scene, params, "metallic.value")

# specular spec_trans=1
params['Material.base_color.value'] = np.array([250, 10, 10]) / 256
params['Material.metallic.value'] = 0.0
params['Material.specular'] = 0.0
images_st_specular = create_images(scene, params, "specular")
# spec_tint spec_trans=1
images_st_spec_tint = create_images(scene, params, "spec_tint.value")

# roughness spec_trans=1
params['Material.base_color.value'] = np.array([46, 47, 99]) / 256
params['Material.roughness.value'] = 0.01
params['Material.spec_tint.value'] = 0.0
params['Material.specular'] = 0.3
images_st_roughness = create_images(scene, params, "roughness.value")

# anisotropic spec_trans=1
params['Material.base_color.value'] = np.array([193, 32, 100]) / 256
params['Material.roughness.value'] = 0.2
params['Material.anisotropic.value'] = 0.0
params['Material.specular'] = 0.4
images_st_anisotropic = create_images(scene, params, "anisotropic.value")

# 2nd scene for regular
scene = mi.load_file('scene_studio.xml')
params = mi.traverse(scene)

# clearcoat
params['Material.base_color.value'] = np.array([10, 40, 200]) / 256
params['Material.roughness.value'] = 0.7
params['Material.clearcoat_gloss.value'] = 1.0
params['Material.specular'] = 1.0
images_clearcoat = create_images(scene, params, "clearcoat.value")
# clearcoat_gloss
images_clearcoat_gloss = create_images(scene, params, "clearcoat_gloss.value")

# clearcoat spec_trans=1
params['Material.base_color.value'] = np.array([30, 100, 200]) / 256
params['Material.roughness.value'] = 0.3
params['Material.clearcoat_gloss.value'] = 1.0
params['Material.spec_trans.value'] = 1.0
params['Material.specular'] = 0.2
images_st_clearcoat = create_images(scene, params, "clearcoat.value")
# clearcoat_gloss spec_trans=1
images_st_clearcoat_gloss = create_images(
    scene, params, "clearcoat_gloss.value")

# 3rd scene for regular
scene = mi.load_file('scene_light.xml')
params = mi.traverse(scene)

# sheen
params['Material.base_color.value'] = np.array([40, 1, 1]) / 256
params['Material.roughness.value'] = 0.9
params['Material.anisotropic.value'] = 0.3
params['Material.specular'] = 0.1
images_sheen = create_images(scene, params, "sheen.value")
# sheen_tint
images_sheen_tint = create_images(scene, params, "sheen_tint.value")

# sheen spec_trans=0.5
params['Material.base_color.value'] = np.array([40, 1, 1]) / 256
params['Material.roughness.value'] = 0.8
params['Material.anisotropic.value'] = 0.0
params['Material.spec_tint.value'] = 1.0
params['Material.sheen_tint.value'] = 0.0
params['Material.spec_trans.value'] = 0.5
params['Material.specular'] = 0.01
images_st_sheen = create_images(scene, params, "sheen.value")
# sheen_tint spec_trans=0.5
images_st_sheen_tint = create_images(scene, params, "sheen_tint.value")

# Output the result
images_normal = []
images_normal.append(images_roughness)
images_normal.append(images_metallic)
images_normal.append(images_specular)
images_normal.append(images_specular_tint)
images_normal.append(images_anisotropic)
images_normal.append(images_clearcoat)
images_normal.append(images_clearcoat_gloss)
images_normal.append(images_sheen)
images_normal.append(images_sheen_tint)
images_normal.append(images_flatness)

# images without transmission
pad = 1
parameters = [
    "roughness",
    "metallic",
    "specular",
    "specular_tint",
    "anisotropic",
    "clearcoat",
    "clearcoat_gloss",
    "sheen",
    "sheen_tint",
    "flatness"]
fig, axs = plt.subplots(12, len(parameters), figsize=(10, 12), dpi=300)

for i in range(len(parameters)):
    axs[0, i].set_title(str(parameters[i]), fontweight='semibold', fontsize=7)

for j in range(11):
    axs[j,
        0].annotate(str(j / 10),
                    xy=(0,
                    0.5),
                    xytext=(-axs[j,
                                 0].yaxis.labelpad - pad,
                            0),
                    xycoords=axs[j,
                                 0].yaxis.label,
                    textcoords='offset points',
                    size='large',
                    ha='right',
                    va='center',
                    fontweight='semibold')
axs[11, 0].annotate("Diff.", xy=(0, 0.5), xytext=(-axs[11, 0].yaxis.labelpad - pad, 0),
                    xycoords=axs[11, 0].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', fontweight='bold', color="red")

for i in range(12):
    for j in range(10):
        axs[i, j].imshow(images_normal[j][i])
        axs[i, j].set_yticks([])
        axs[i, j].set_xticks([])
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("principled_blend.png", bbox_inches='tight', pad_inches=0.01)

# images with transmission
images_st = []
images_st.append(images_st_roughness)
images_st.append(images_st_metallic)
images_st.append(images_st_specular)
images_st.append(images_st_spec_tint)
images_st.append(images_st_anisotropic)
images_st.append(images_st_clearcoat)
images_st.append(images_st_clearcoat_gloss)
images_st.append(images_st_sheen)
images_st.append(images_st_sheen_tint)
images_st.append(images_strans)

parameters = [
    "roughness",
    "metallic",
    "specular",
    "specular_tint",
    "anisotropic",
    "clearcoat",
    "clearcoat_gloss",
    "sheen",
    "sheen_tint",
    "spec_trans"]
fig, axs = plt.subplots(12, len(parameters), figsize=(10, 12), dpi=300)

for i in range(len(parameters)):
    axs[0, i].set_title(str(parameters[i]), fontweight='semibold', fontsize=7)

for j in range(11):
    axs[j,
        0].annotate(str(j / 10),
                    xy=(0,
                    0.5),
                    xytext=(-axs[j,
                                 0].yaxis.labelpad - pad,
                            0),
                    xycoords=axs[j,
                                 0].yaxis.label,
                    textcoords='offset points',
                    size='large',
                    ha='right',
                    va='center',
                    fontweight='semibold')
axs[11, 0].annotate("Diff.", xy=(0, 0.5), xytext=(-axs[11, 0].yaxis.labelpad - pad, 0),
                    xycoords=axs[11, 0].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', fontweight='bold', color="red")

for i in range(12):
    for j in range(10):
        axs[i, j].imshow(images_st[j][i])
        axs[i, j].set_yticks([])
        axs[i, j].set_xticks([])
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("principled_st_blend.png", bbox_inches='tight', pad_inches=0.01)
