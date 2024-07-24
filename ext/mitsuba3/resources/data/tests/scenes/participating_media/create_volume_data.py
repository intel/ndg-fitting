import numpy as np


def write_binary_grid3d(filename, values):
    with open(filename, 'wb') as f:
        f.write(b'V')
        f.write(b'O')
        f.write(b'L')
        f.write(np.uint8(3).tobytes())  # Version
        f.write(np.int32(1).tobytes())  # type
        f.write(np.int32(values.shape[0]).tobytes())  # size
        f.write(np.int32(values.shape[1]).tobytes())
        f.write(np.int32(values.shape[2]).tobytes())
        if values.ndim == 3:
            f.write(np.int32(1).tobytes())  # channels
        else:
            f.write(np.int32(values.shape[3]).tobytes())  # channels
        f.write(np.float32(0.0).tobytes())  # bbox
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(values.ravel().astype(np.float32).tobytes())


res = 16

sigmat = np.zeros((res, res, res))
sigmat += np.maximum(0, (np.sin(np.linspace(0, 16, res)) + 1) * 0.5)
sigmat *= np.linspace(0, 1, res)


def sigmat_f(x, y, z):
    if x > 0.2 and y > 0.7 and z < 0.5:
        return 5
    return np.maximum(0, (np.sin(16 * (x + 4*((y - 0.5) ** 2 - 0.25)) + 1) * 0.5))


for z in range(res):
    for y in range(res):
        for x in range(res):
            sigmat[z, y, x] = sigmat_f(x / res, y / res, z / res)

sigmat = np.maximum(sigmat, 0)
write_binary_grid3d('textures/sigmat.vol', sigmat)

albedo = np.zeros((res, res, res, 3))


def albedo_f(x, y, z):
    return np.cos(45 * (x + ((y - 0.7) ** 2 - 0.25)) + 1) * 0.5


for z in range(res):
    for y in range(res):
        for x in range(res):
            albedo[z, y, x, 0] = albedo_f(x / res, y / res, z / res) + 0.5 * np.cos(y / res) + 0.5
            albedo[z, y, x, 1] = sigmat_f(((x / res) + 0.3) % 1, ((y / res)+0.7) % 1, z / res)
            albedo[z, y, x, 2] = albedo_f(x / res, y / res, z / res)

albedo = np.clip(albedo, 0, 1)
write_binary_grid3d('textures/albedo.vol', albedo)
