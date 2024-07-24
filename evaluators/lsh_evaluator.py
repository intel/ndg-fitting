# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: MIT License

import torch
import taichi as ti

# For now the dimensionality of the kernels need to be defined explicitly before training
g_dim = 10

feature_type = ti.types.vector(n=3, dtype=ti.f32)
vector_type = ti.types.vector(n=g_dim, dtype=ti.f32)
matrix_type = ti.types.matrix(n=g_dim, m=g_dim, dtype=ti.f32)


@ti.kernel
def cull_gs_lsh(
        q_projections: ti.types.ndarray(ti.float32, ndim=2),
        g_projections: ti.types.ndarray(ti.float32, ndim=2),
        g_projections_range: ti.types.ndarray(ti.float32, ndim=2),
        n_gs: ti.i32,
        n_rounds: ti.i32,
        out_mask: ti.types.ndarray(ti.int32, ndim=1),
):
    for g_id in range(n_gs):
        q_id = g_id

        for r_id in range(n_rounds):
            g_min = g_projections[g_id, r_id] - g_projections_range[g_id, r_id]/2.0
            g_max = g_projections[g_id, r_id] + g_projections_range[g_id, r_id]/2.0

            if not (g_min <= q_projections[q_id, r_id] <= g_max):
                out_mask[q_id] = 0
                break


@ti.kernel
def evaluate_gs_forward_lsh(
        q: ti.types.ndarray(vector_type, ndim=1),
        q_projections: ti.types.ndarray(ti.float32, ndim=2),
        g: ti.types.ndarray(vector_type, ndim=1),
        g_projections: ti.types.ndarray(ti.float32, ndim=2),
        g_projections_range: ti.types.ndarray(ti.float32, ndim=2),
        tile_ranges: ti.types.ndarray(ti.float32, ndim=3),
        tile_g_ids: ti.types.ndarray(ti.int32, ndim=2),
        tile_g_counts: ti.types.ndarray(ti.int32, ndim=1),
        v_inv: ti.types.ndarray(matrix_type, ndim=1),
        fs: ti.types.ndarray(feature_type, ndim=1),
        n_pixels: ti.i32,
        n_gs: ti.i32,
        n_rounds: ti.i32,
        n_tiles: ti.i32,
        out: ti.types.ndarray(feature_type, ndim=1),
        out_tile_evals: ti.types.ndarray(ti.int32, ndim=1),
        out_evals: ti.types.ndarray(ti.int32, ndim=1),
        out_ids: ti.types.ndarray(ti.int32, ndim=2),
        out_pdfs: ti.types.ndarray(ti.float32, ndim=2)
):
    ti.loop_config(block_dim=256)
    for pixel_id in range(n_pixels):
        res: ti.i32 = n_pixels ** 0.5

        pixel_row = pixel_id // res
        pixel_column = pixel_id % res

        # Calculate the row and column of the tile
        tile_row = pixel_row // 16
        tile_column = pixel_column // 16

        # Calculate number of tiles per row
        tiles_per_row = res // 16

        # Calculate the tile ID
        tile_id = tile_row * tiles_per_row + tile_column

        for r_id in range(n_rounds):
            ti.atomic_max(tile_ranges[tile_id, r_id, 1], q_projections[pixel_id, r_id])
            ti.atomic_min(tile_ranges[tile_id, r_id, 0], q_projections[pixel_id, r_id])

    for g_id in range(n_gs):

        for tile_id in range(n_tiles):

            culled = False

            for r_id in range(n_rounds):
                tile_min = tile_ranges[tile_id, r_id, 0]
                tile_max = tile_ranges[tile_id, r_id, 1]

                g_min = g_projections[g_id, r_id] - g_projections_range[g_id, r_id]/2.0
                g_max = g_projections[g_id, r_id] + g_projections_range[g_id, r_id]/2.0

                if not (g_min <= tile_max and tile_min <= g_max):
                    culled = True
                    break

            if not culled:
                offset = ti.atomic_add(tile_g_counts[tile_id], 1)
                ti.atomic_add(out_tile_evals[tile_id], 1)
                ti.atomic_add(tile_g_ids[tile_id, offset], g_id)

    ti.loop_config(block_dim=256)
    for pixel_id in range(n_pixels):
        res: ti.i32 = n_pixels ** 0.5

        pixel_row = pixel_id // res
        pixel_column = pixel_id % res

        # Calculate the row and column of the tile
        tile_row = pixel_row // 16
        tile_column = pixel_column // 16

        # Calculate number of tiles per row
        tiles_per_row = res // 16

        # Calculate the tile ID
        tile_id = tile_row * tiles_per_row + tile_column

        val = feature_type(0.0)

        evals = 0

        for g_count in range(tile_g_counts[tile_id]):
            g_id = tile_g_ids[tile_id, g_count]

            evals += 1

            x = q[pixel_id] - g[g_id]

            pdf = ti.exp(-0.5 * ti.abs(x @ v_inv[g_id] @ x))

            val += pdf * fs[g_id]

            for k in range(10):
                if pdf > out_pdfs[pixel_id, k]:
                    out_pdfs[pixel_id, k] = pdf
                    out_ids[pixel_id, k] = g_id
                    break

        out_evals[pixel_id] = evals

        out[pixel_id] = val


@ti.kernel
def evaluate_gs_backward_lsh(
        grad_out: ti.types.ndarray(feature_type, ndim=1),
        q: ti.types.ndarray(vector_type, ndim=1),
        g: ti.types.ndarray(vector_type, ndim=1),
        v_inv: ti.types.ndarray(matrix_type, ndim=1),
        out_ids: ti.types.ndarray(ti.int32, ndim=2),
        out_pdfs: ti.types.ndarray(ti.float32, ndim=2),
        fs: ti.types.ndarray(feature_type, ndim=1),
        n_pixels: ti.i32,
        d_fs: ti.types.ndarray(feature_type, ndim=1),
        d_m: ti.types.ndarray(vector_type, ndim=1),
        d_v_inv: ti.types.ndarray(matrix_type, ndim=1),
):
    for pixel_id in range(n_pixels):

        for top_id in range(10):
            g_id = out_ids[pixel_id, top_id]

            if g_id < 0:
                break

            x = q[pixel_id] - g[g_id]

            pdf = out_pdfs[pixel_id, top_id]

            d_fs[g_id] += grad_out[pixel_id] * pdf

            d_m[g_id] += grad_out[pixel_id] @ fs[g_id] * pdf * (v_inv[g_id] @ x)

            d_v_inv[g_id] += grad_out[pixel_id] @ fs[g_id] * pdf * (-0.5) * x.outer_product(x)


class EvaluatorLSH(torch.autograd.Function):

    @staticmethod
    def cull(q_projections, m_projections, m_projections_range):
        n_gs = m_projections.shape[0]
        n_rounds = m_projections.shape[1]

        out_mask = torch.ones([n_gs], device=q_projections.device, dtype=torch.int32)

        cull_gs_lsh(q_projections, m_projections, m_projections_range, n_gs, n_rounds, out_mask)

        return out_mask

    @staticmethod
    def forward(ctx, q, q_projections, m, m_projections, m_projections_range, v_inv, fs, max_tiles):
        n_pixels = q.shape[0]
        n_gs = m.shape[0]
        n_rounds = m_projections.shape[1]
        n_tiles = n_pixels//(16**2)
        tile_ranges = torch.zeros([n_tiles, n_rounds, 2], device=q.device, dtype=torch.float32)
        tile_ranges[:, :, 0] += 1e4
        tile_ranges[:, :, 1] -= 1e4
        tile_g_ids = torch.zeros([n_tiles, max_tiles], device=q.device, dtype=torch.int32)
        tile_g_counts = torch.zeros([n_tiles], device=q.device, dtype=torch.int32)
        out = torch.zeros([q.shape[0], fs.shape[1]], device=q.device)
        out_tile_evals = torch.zeros([n_tiles], device=q.device, dtype=torch.int32)
        out_evals = torch.zeros([q.shape[0]], device=q.device, dtype=torch.int32)
        out_ids = torch.zeros([q.shape[0], 10], device=q.device, dtype=torch.int32)-1
        out_pdfs = torch.zeros([q.shape[0], 10], device=q.device, dtype=torch.float32)

        evaluate_gs_forward_lsh(q, q_projections, m, m_projections, m_projections_range, tile_ranges.contiguous(), tile_g_ids.contiguous(), tile_g_counts.contiguous(), v_inv, fs, n_pixels, n_gs, n_rounds, n_tiles, out.contiguous(), out_tile_evals.contiguous(), out_evals.contiguous(), out_ids.contiguous(), out_pdfs.contiguous())

        ctx.save_for_backward(q, m, v_inv, out_ids, out_pdfs, fs)

        return out, out_tile_evals, out_evals, out_ids

    @staticmethod
    def backward(ctx, grad_out, grad_out_tile_evals, grad_out_evals, grad_out_ids):
        q, m, v_inv, out_ids, out_pdfs, fs = ctx.saved_tensors

        d_q = torch.zeros_like(q)
        d_m = torch.zeros_like(m)
        d_v_inv = torch.zeros_like(v_inv)
        d_fs = torch.zeros_like(fs)

        evaluate_gs_backward_lsh(grad_out, q, m, v_inv, out_ids, out_pdfs, fs, q.shape[0], d_fs, d_m, d_v_inv)

        return d_q, None, d_m, None, None, d_v_inv, d_fs, None
