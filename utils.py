import numpy as np
import os
import math
import torch
import cv2
import glfw
import OpenGL.GL as gl

# OpenGL & GLFW


def bind_texture(texture_img, resolution):
    texture_id = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB16F, resolution[0], resolution[1], 0, gl.GL_RGB, gl.GL_FLOAT, texture_img)

    return texture_id


def replace_texture(texture_img, texture_id, resolution, exposure=1.0):
    texture_img = texture_img.detach()[0, :, :, :] * exposure
    texture_img = texture_img.float().cpu().numpy()
    texture_img = cv2.resize(texture_img, (resolution[0], resolution[1]), interpolation=cv2.INTER_NEAREST)

    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, resolution[0], resolution[1], gl.GL_RGB, gl.GL_FLOAT, texture_img)


def impl_glfw_init(width, height, window_name):
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


def load_tensor_from_exr(filename, target_resolution=None):
    image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # OpenCV loads in BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if target_resolution is not None:
        image = cv2.resize(image, target_resolution, interpolation=cv2.INTER_AREA)

    # OpenCV has row major convention
    image = image.transpose(1, 0, 2)
    image = torch.from_numpy(image)

    return image


def create_3d_grid(resolution):
    x, y, z = torch.meshgrid([torch.linspace(0, 1, resolution[0]), torch.linspace(0, 1, resolution[1]), torch.linspace(0, 1, resolution[2])])
    grid = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)

    return grid


def create_2d_grid(resolution, start=0, end=0):
    x, y = torch.meshgrid([torch.linspace(start, end, resolution[0]), torch.linspace(start, end, resolution[1])])
    grid = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)

    return grid


def print_params(self):
    print("Number of model parameters:")
    params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("%d" % params)


def get_params(variables, custom_values):
    params = []

    for i in range(len(variables)):
        # Don't inform the network of the camera params
        if 'sensor' in variables[i].id():
            continue
        for j in range(variables[i].num_parameters()):
            params.append(custom_values[i][j])

    return params


def stack_inputs_tensor(buffers, variables, custom_values):
    resolution = (buffers[0].shape[0], buffers[0].shape[1], 1)

    variable_buffers = []

    for i in range(len(variables)):
        # Don't inform the network of the camera params
        if 'sensor' in variables[i].id():
            continue
        for j in range(variables[i].num_parameters()):
            variable_buffers.append(torch.full(resolution, custom_values[i][j] * 2.0 - 1.0, device='cuda'))

    inputs = torch.cat([*buffers, *variable_buffers], 2)

    return inputs


def create_custom_values_tensor(variables, custom_values):
    res = torch.tensor([])

    for i in range(len(variables)):
        for j in range(variables[i].num_parameters()):
            res = torch.cat((res, torch.tensor(custom_values[i][j]).unsqueeze(0)))

    return res


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def linear_to_srgb(x, gamma=2.4):
    x = torch.where(x <= 0.0031308, x * 12.92, 1.055 * abs(x) ** (1 / gamma) - 0.055)

    return x


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def softplus(x, b=1.0):
    return torch.log(1 + torch.exp(b * x)) / b


def inverse_softplus(x, b=1.0):
    return torch.log(torch.exp(b * x) - 1) / b


def create_diag(diag):
    B, N = diag.shape

    L = torch.zeros(B, N, N, dtype=diag.dtype, device=diag.device)
    L.view(B, -1)[:, ::N + 1] = diag

    return L


def create_cholesky(diag, l_triang):
    L = create_triang(diag, l_triang)

    symm_matrix = torch.bmm(L, L.transpose(-1, -2))

    return symm_matrix


def compute_lr(step, lr_init, lr_final, max_steps=30000):
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return log_lerp


def create_cholesky_matrix(L):
    B, N, N = L.shape

    symm_matrix = torch.bmm(L, L.transpose(-1, -2))

    return symm_matrix


def get_cholesky(symm_matrix):
    B, N, N = symm_matrix.shape

    diag = symm_matrix[:, torch.arange(N), torch.arange(N)]

    s_inv = torch.zeros(B, N, N, dtype=diag.dtype, device=diag.device)
    s_inv.view(B, -1)[:, ::N + 1] = 1.0/(diag+1e-6)

    symm_matrix = torch.bmm(s_inv, symm_matrix)

    l_triang = symm_matrix[:, torch.tril_indices(N, N, offset=-1)[0], torch.tril_indices(N, N, offset=-1)[1]]

    return diag, l_triang


def create_triang(diag, l_triang):
    B, N = diag.shape

    L = torch.eye(N, dtype=diag.dtype, device=diag.device).unsqueeze(0).repeat(B, 1, 1)

    L[:, torch.tril_indices(N, N, offset=-1)[0], torch.tril_indices(N, N, offset=-1)[1]] = l_triang

    S = create_diag(diag)

    L = S @ L

    return L


def sample_from_gs(m, diags, l_triangs, n_samples):
    L = create_triang(diags, l_triangs).unsqueeze(1).repeat(1, n_samples, 1, 1)
    samples = torch.randn([m.shape[0], n_samples, m.shape[-1]], device=m.device)
    samples = torch.einsum('ijk,ijkm->ijk', samples, L) + m.unsqueeze(1).repeat(1, n_samples, 1)

    return samples


def copy_diag(matrix):
    B, N, _ = matrix.shape

    L = torch.zeros(B, N, N, dtype=matrix.dtype, device=matrix.device)
    L.view(B, -1)[:, ::N + 1] = matrix.view(B, -1)[:, ::N + 1]

    return L


def copy_l_triang(matrix):
    B, N, _ = matrix.shape

    L = torch.zeros(B, N, N, dtype=matrix.dtype, device=matrix.device)

    L[:, torch.tril_indices(N, N, offset=-1)[0], torch.tril_indices(N, N, offset=-1)[1]] = matrix[:, torch.tril_indices(N, N, offset=-1)[0], torch.tril_indices(N, N, offset=-1)[1]]

    return L


def matrix_to_list(mat):
    return list(map(list, list(mat)))


def update_learning_rate(optimizer, lr, param_name):
    for param_group in optimizer.param_groups:
        if param_group["name"] == param_name:
            param_group['lr'] = lr


def mask_optimizer(optimizer, mask, group_names):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if group.get('name') in group_names:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def remove_group_optimizer(optimizer, group_names):
    # Find the index of the parameter group with the specified name
    index_to_remove = None
    for i, group in enumerate(optimizer.param_groups):
        if group.get('name') in group_names:
            index_to_remove = i
            break

    # Remove the parameter group if found
    if index_to_remove is not None:
        optimizer.param_groups.pop(index_to_remove)


def cat_optimizer(optimizer, tensors_dict):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        if group["name"] in tensors_dict:
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors


def normalize_vecs_np(vector):
    return vector / (np.linalg.norm(vector, axis=-1, keepdims=True))


def normalize_vecs_torch(vector):
    return vector / (torch.norm(vector, dim=-1, keepdim=True))


def translation_matrix(translation):
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return matrix


def rotation_matrix(axis, theta):
    theta = theta * np.pi / 180.0
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                     [0, 0, 0, 1]])


def camera_to_world(origin, target, up):
    forward = normalize_vecs_np(target - origin)

    right = -normalize_vecs_np(np.cross(up, forward))
    up = normalize_vecs_np(np.cross(right, forward))

    rotation = np.eye(4)
    rotation[:3, :3] = np.stack((right, up, forward), axis=-1)

    translation = translation_matrix(origin)

    transformation = (translation @ rotation)

    return transformation


# TODO: cleanup
def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)


def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def hash_vectors_cosine(n_buckets, vecs, rotations, n_hashes=8):
    batch_size = vecs.shape[0]
    device = vecs.device

    # Normalize for LSH
    norm_rotations = rotations / rotations.norm(dim=0, keepdim=True)
    norm_vecs = vecs / vecs.norm(dim=-1, keepdim=True)

    assert n_buckets % 2 == 0

    rot_size = n_buckets

    rotated_vecs = torch.mm(norm_vecs, norm_rotations)

    rotated_vecs = rotated_vecs.reshape(-1, n_hashes, rot_size // 2).permute(1, 0, 2)

    rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
    buckets = torch.argmax(rotated_vecs, dim=-1).type(torch.int32).permute(1, 0)

    return buckets


def project_vectors_gaussians(vecs, projection_vecs, cov=None, n_hashes=8):
    vecs = vecs.unsqueeze(0).repeat(n_hashes, 1, 1)

    projections = torch.einsum("ijk,ik->ji", vecs, projection_vecs)
    projections_range = torch.zeros_like(projections)

    # Compute range for gaussians
    if cov is not None:
        cov = cov.unsqueeze(0).repeat(n_hashes, 1, 1, 1)
        projection_vecs = projection_vecs.unsqueeze(1).unsqueeze(-1).repeat(1, vecs.shape[1], 1, 1)

        eigen_projections = torch.einsum("ijum,ijmk->ijuk", projection_vecs.permute(0, 1, 3, 2), cov)
        eigen_projections = torch.einsum("ijum,ijmu->ij", eigen_projections, projection_vecs)

        projections_range = (3.0 * torch.sqrt(torch.abs(eigen_projections))).permute(1, 0)

    return projections, projections_range
