import glob
import os
from os.path import dirname, join, realpath

# This script can be used to either view the reference images or the rendered images in a single grid layout

import numpy as np

import mitsuba

mitsuba.set_variant('scalar_rgb')

from mitsuba.core import Bitmap

def main():
    current_dir = realpath(dirname(__file__))
    exr_files = sorted(glob.glob(join(current_dir, 'refs', '*_ref_*.exr')))

    images = []
    for f in exr_files:
        if (os.path.basename(f) == 'result.exr'):
            continue
        images.append(np.array(Bitmap(f)))

    images_np = np.stack(images, axis=-1)
    s = images_np.shape
    h = s[0]
    w = s[1]
    images_np = np.swapaxes(images_np, 2, 3)
    print(f"images_np.shape: {images_np.shape}")
    images_np = np.swapaxes(images_np, 0, 2)
    images_np = np.swapaxes(images_np, 1, 2)
    print(f"images_np.shape: {images_np.shape}")

    n_cols = 4
    n_rows = len(images) // n_cols
    images_np = np.reshape(images_np, [n_rows, n_cols, h, w, 3])
    images_np = np.swapaxes(images_np, 1, 2)
    images_np = np.reshape(images_np, [h*n_rows, w*n_cols, 3])

    # images_np = np.reshape(images_np, [h * 12, w * 2, 3])
    print(f"images_np.shape: {images_np.shape}")
    Bitmap(images_np).write('result.exr')


if __name__ == '__main__':
    main()
