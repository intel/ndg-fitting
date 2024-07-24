# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: MIT License

import torch.nn as nn
import taichi as ti

from utils import *
from evaluators.lsh_evaluator import EvaluatorLSH
from torch_ema import ExponentialMovingAverage
from plyfile import PlyData, PlyElement

main_gs_color = [0, 0.4, 1]
sub_gs_color = [0, 1, 0]


class GaussianGenerator(nn.Module):

    def __init__(self, dimensions, init_gs, ws_act, ws_act_inv, fs_act, fs_act_inv, ws_lr, fs_lr, m_sb_lr, n_projection_vectors, cov_bias=1e-1, init_ws=1e-3, init_fs=1.0, init_fs_sb=1.0, fs_dim=3, betas=(0.9, 0.999), device='cuda'):
        super(GaussianGenerator, self).__init__()

        ti.init(arch=ti.cuda, device_memory_GB=0.1)

        self.cov_bias = cov_bias
        self.n_gs_div = 2

        init_n_gs = init_gs.shape[0]

        # Dimensions
        self.device = device
        self.gs_dim = dimensions
        self.diags_dim = dimensions
        self.l_triangs_dim = dimensions * (dimensions - 1) // 2
        self.fs_dim = fs_dim
        self.cs_dim = 3
        self.ts_dim = 3

        # Initialization params
        self.init_ws = init_ws
        self.init_fs = init_fs
        self.init_fs_sb = init_fs_sb

        # Activation functions
        self.ws_act = ws_act
        self.ws_act_inv = ws_act_inv
        self.fs_act = fs_act
        self.fs_act_inv = fs_act_inv

        self.diags_act = lambda x: torch.exp(x)
        self.diags_act_inv = lambda x: torch.log(torch.abs(x+1e-6))
        self.l_triangs_act = lambda x: torch.sigmoid(x)*2.0-1.0
        self.l_triangs_act_inv = lambda x: inverse_sigmoid(torch.clip((x+1.0)/2.0, min=1e-6, max=1.0 - 1e-6))

        # Learning rates
        self.m_lr = 1e-3
        self.diags_lr = 1e-2
        self.l_triangs_lr = 1e-2
        self.ws_lr = ws_lr
        self.fs_lr = fs_lr

        self.m_sb_lr = m_sb_lr

        self.m_lr_cur = self.m_lr
        self.diags_lr_cur = self.diags_lr
        self.l_triangs_lr_cur = self.l_triangs_lr

        # Main Gaussians start without sub Gaussians
        self.seed_indices = torch.zeros([0], dtype=torch.int, device=device)

        # Main Gaussians
        self.m = torch.nn.Parameter(init_gs)
        self.diags = torch.nn.Parameter(self.diags_act_inv(torch.ones([init_n_gs, self.gs_dim], device=device) * self.cov_bias))
        self.l_triangs = torch.nn.Parameter(self.l_triangs_act_inv(torch.zeros([init_n_gs, self.gs_dim*(self.gs_dim-1)//2], device=device)))
        self.ws = torch.nn.Parameter(self.ws_act_inv(torch.ones([init_n_gs, 1], device=device) * init_ws))
        self.fs = torch.nn.Parameter(self.fs_act_inv(torch.ones([init_n_gs, self.fs_dim], device=device) * init_fs))
        self.cs = normalize_vecs_torch(torch.rand([init_n_gs, 3], device=device))
        self.ts = torch.tensor([main_gs_color], dtype=torch.float, device=device).repeat(init_n_gs, 1)

        # Sub Gaussians
        self.m_sb = None
        self.diags_sb = None
        self.l_triangs_sb = None
        self.ws_sb = None
        self.fs_sb = None
        self.cs_sb = None
        self.ts_sb = None

        # LSH Culling params
        self.n_projection_vecs = n_projection_vectors

        self.projection_vecs = torch.randn(self.n_projection_vecs, self.gs_dim, device=device)
        self.projection_vecs /= self.projection_vecs.norm(dim=-1, keepdim=True)

        self.out_ids = None

        params = [
            {'params': [self.m], 'lr': self.m_lr, "name": "m"},
            {'params': [self.diags], 'lr': self.diags_lr, "name": "diags"},
            {'params': [self.l_triangs], 'lr': self.l_triangs_lr, "name": "l_triangs"},
            {'params': [self.ws], 'lr': self.ws_lr, "name": "ws"},
            {'params': [self.fs], 'lr': self.fs_lr, "name": "fs"},
        ]

        self.optimizer = torch.optim.Adam(params, betas=betas, eps=1e-15)
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.995)

    @property
    def n_gs(self):
        return self.m.shape[0]

    @property
    def n_gs_sb(self):
        return self.seed_indices.shape[0]

    @property
    def gs_ids(self):
        return torch.arange(0, self.n_gs, device=self.device)

    def update_lr(self, step):
        self.m_lr_cur = compute_lr(step, lr_init=self.m_lr, lr_final=self.m_lr*0.01)
        self.diags_lr_cur = compute_lr(step, lr_init=self.diags_lr, lr_final=self.diags_lr*0.01)
        self.l_triangs_lr_cur = compute_lr(step, lr_init=self.l_triangs_lr, lr_final=self.l_triangs_lr*0.01)

        update_learning_rate(self.optimizer, self.m_lr_cur, param_name='m')
        update_learning_rate(self.optimizer, self.diags_lr_cur, param_name='diags')
        update_learning_rate(self.optimizer, self.l_triangs_lr_cur, param_name='l_triangs')
        update_learning_rate(self.optimizer, self.diags_lr_cur, param_name='diags_sb')
        update_learning_rate(self.optimizer, self.l_triangs_lr_cur, param_name='l_triangs_sb')

    def get_non_seeded_gs_mask(self):
        gs_ids = torch.arange(0, self.n_gs, device=self.device)

        return ~torch.isin(gs_ids, self.seed_indices)

    def compute_gs_sb_abs(self, mask):
        assert self.n_gs_sb > 0
        assert mask.nonzero().shape[0] > 0

        L = create_triang(self.diags_act(self.diags[self.seed_indices][mask]), self.l_triangs_act(self.l_triangs[self.seed_indices][mask]))
        M = create_triang(self.diags_act(self.diags_sb[mask]), self.l_triangs_act(self.l_triangs_sb[mask]))

        LM = torch.bmm(L, M)
        diags_sb_abs, l_triangs_sb_abs = get_cholesky(LM)

        diags_sb_abs = self.diags_act_inv(diags_sb_abs)
        l_triangs_sb_abs = self.l_triangs_act_inv(l_triangs_sb_abs)

        m_sb_abs = torch.einsum('ijk, ij->ij', L, self.m_sb[mask]) + self.m[self.seed_indices][mask]
        ws_sb_abs = self.ws_sb[mask]
        fs_sb_abs = self.fs_sb[mask]
        cs_sb_abs = self.cs_sb[mask]
        ts_sb_asb = self.ts_sb[mask]

        return m_sb_abs, diags_sb_abs, l_triangs_sb_abs, ws_sb_abs, fs_sb_abs, cs_sb_abs, ts_sb_asb

    def create_gs_sb_from(self, indices):
        n_gs_sb = indices.shape[0]

        assert n_gs_sb > 0

        m_sb = torch.zeros_like(self.m[indices])
        diags_sb = self.diags_act_inv(torch.ones([n_gs_sb, self.diags_dim], device=self.device))
        l_triangs_sb = self.l_triangs_act_inv(torch.zeros([n_gs_sb, self.l_triangs_dim], device=self.device))
        ws_sb = self.ws_act_inv(torch.ones([n_gs_sb, 1], device=self.device) * self.init_ws)
        fs_sb = self.fs_act_inv(torch.rand([n_gs_sb, self.fs_dim], device=self.device) * self.init_fs_sb)
        cs_sb = normalize_vecs_torch(torch.rand([n_gs_sb, self.cs_dim], device=self.device))
        ts_sb = torch.tensor([sub_gs_color], device=self.device).repeat(n_gs_sb, 1)

        return m_sb, diags_sb, l_triangs_sb, ws_sb, fs_sb, cs_sb, ts_sb

    def optimizer_create_gs_sb(self, m_sb, diags_sb, l_triangs_sb, ws_sb, fs_sb, cs_sb, ts_sb):
        self.m_sb = torch.nn.Parameter(m_sb.clone().detach())
        self.diags_sb = torch.nn.Parameter(diags_sb.clone().detach())
        self.l_triangs_sb = torch.nn.Parameter(l_triangs_sb.clone().detach())
        self.ws_sb = torch.nn.Parameter(ws_sb.clone().detach())
        self.fs_sb = torch.nn.Parameter(fs_sb.clone().detach())
        self.cs_sb = cs_sb.clone().detach()
        self.ts_sb = ts_sb.clone().detach()

        m_params = {'params': self.m_sb, 'lr': self.m_sb_lr, 'name': 'm_sb'}
        diags_params = {'params': self.diags_sb, 'lr': self.diags_lr_cur, 'name': 'diags_sb'}
        l_triangs_params = {'params': self.l_triangs_sb, 'lr': self.l_triangs_lr_cur, 'name': 'l_triangs_sb'}
        ws_params = {'params': self.ws_sb, 'lr': self.ws_lr, 'name': 'ws_sb'}
        fs_params = {'params': self.fs_sb, 'lr': self.fs_lr, 'name': 'fs_sb'}

        self.optimizer.add_param_group(m_params)
        self.optimizer.add_param_group(diags_params)
        self.optimizer.add_param_group(l_triangs_params)
        self.optimizer.add_param_group(ws_params)
        self.optimizer.add_param_group(fs_params)

        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.995)

    def optimizer_concat_gs_sb(self, m_sb, diags_sb, l_triangs_sb, ws_sb, fs_sb, cs_sb, ts_sb):
        gs_sb_dict = {"m_sb": m_sb,
                     "diags_sb": diags_sb,
                     "l_triangs_sb": l_triangs_sb,
                     "ws_sb": ws_sb,
                     "fs_sb": fs_sb}

        optimizable_tensors = cat_optimizer(self.optimizer, gs_sb_dict)

        self.m_sb = optimizable_tensors["m_sb"]
        self.diags_sb = optimizable_tensors["diags_sb"]
        self.l_triangs_sb = optimizable_tensors["l_triangs_sb"]
        self.ws_sb = optimizable_tensors["ws_sb"]
        self.fs_sb = optimizable_tensors["fs_sb"]
        self.cs_sb = torch.cat([self.cs_sb, cs_sb], dim=0)
        self.ts_sb = torch.cat([self.ts_sb, ts_sb], dim=0)

        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.995)

    def optimizer_concat_gs(self, m, diags, l_triangs, ws, fs, cs, ts):
        gs_dict = {"m": m,
                 "diags": diags,
                 "l_triangs": l_triangs,
                 "ws": ws,
                 "fs": fs}

        optimizable_tensors = cat_optimizer(self.optimizer, gs_dict)

        self.m = optimizable_tensors["m"]
        self.diags = optimizable_tensors["diags"]
        self.l_triangs = optimizable_tensors["l_triangs"]
        self.ws = optimizable_tensors["ws"]
        self.fs = optimizable_tensors["fs"]
        self.cs = torch.cat([self.cs, cs], dim=0)
        self.ts = torch.cat([self.ts, ts], dim=0)

        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.995)

    def optimizer_keep_gs(self, mask):
        optimizable_tensors = mask_optimizer(self.optimizer, mask, ['m', 'diags', 'l_triangs', 'ws', 'fs'])

        self.m = optimizable_tensors['m']
        self.diags = optimizable_tensors['diags']
        self.l_triangs = optimizable_tensors['l_triangs']
        self.ws = optimizable_tensors['ws']
        self.fs = optimizable_tensors['fs']
        self.cs = self.cs[mask]
        self.ts = self.ts[mask]

        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.995)

    def optimizer_keep_gs_sb(self, mask):
        optimizable_tensors = mask_optimizer(self.optimizer, mask, ['m_sb', 'diags_sb', 'l_triangs_sb', 'ws_sb', 'fs_sb'])

        self.m_sb = optimizable_tensors['m_sb']
        self.diags_sb = optimizable_tensors['diags_sb']
        self.l_triangs_sb = optimizable_tensors['l_triangs_sb']
        self.ws_sb = optimizable_tensors['ws_sb']
        self.fs_sb = optimizable_tensors['fs_sb']
        self.cs_sb = self.cs_sb[mask]
        self.ts_sb = self.ts_sb[mask]

        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.995)

    def seed_gs(self, threshold):

        non_seeded_gs_ids = self.gs_ids[self.get_non_seeded_gs_mask()]

        seed_mask = self.ws_act(self.ws)[:, 0][non_seeded_gs_ids] > threshold

        if seed_mask.nonzero().shape[0] == 0:
            return

        seed_indices = non_seeded_gs_ids[seed_mask].repeat(self.n_gs_div)

        m_sb, diags_sb, l_triangs_sb, ws_sb, fs_sb, cs_sb, ts_sb = self.create_gs_sb_from(seed_indices)

        if self.n_gs_sb == 0:
            self.optimizer_create_gs_sb(m_sb, diags_sb, l_triangs_sb, ws_sb, fs_sb, cs_sb, ts_sb)
        else:
            self.optimizer_concat_gs_sb(m_sb, diags_sb, l_triangs_sb, ws_sb, fs_sb, cs_sb, ts_sb)

        self.seed_indices = torch.cat([self.seed_indices, seed_indices])
            
    def grow_gs_sb(self, threshold):

        if self.seed_indices.shape[0] == 0:
            return
        
        grow_mask = self.ws_act(self.ws_sb)[:, 0] > threshold

        if grow_mask.nonzero().shape[0] == 0:
            return

        m, diags, l_triangs, ws, fs, cs, ts = self.compute_gs_sb_abs(grow_mask)

        ts = torch.tensor([main_gs_color], dtype=torch.float, device=self.device).repeat(grow_mask.nonzero().shape[0], 1)

        self.optimizer_concat_gs(m.clone().detach(), diags.clone().detach(), l_triangs.clone().detach(), ws.clone().detach(), fs.clone().detach(), cs.clone().detach(), ts.clone().detach())

        if (~grow_mask).nonzero().shape[0] > 0:
            self.optimizer_keep_gs_sb(~grow_mask)

        self.seed_indices = self.seed_indices[~grow_mask]

    def prune_gs(self, threshold):

        keep_mask = self.ws_act(self.ws)[:, 0] >= threshold

        self.optimizer_keep_gs(keep_mask)

    def finalize_gs(self):

        self.grow_gs_sb(0.0)

        self.ema.copy_to()

        self.seed_indices = torch.zeros([0], dtype=torch.int, device=self.device)

        self.m_sb = None
        self.diags_sb = None
        self.l_triangs_sb = None
        self.ws_sb = None
        self.fs_sb = None
        self.cs_sb = None
        self.ts_sb = None

    def get_gs(self, invert=True):
        v = create_cholesky(self.diags_act(self.diags), self.l_triangs_act(self.l_triangs))

        if invert is True:
            v_inv = torch.inverse(v)
        else:
            v_inv = v

        return self.m, v, v_inv, self.ws_act(self.ws), self.fs_act(self.fs), self.cs, self.ts

    def get_gs_sb(self, invert=True):
        m_sb_abs, diags_sb_abs, l_triangs_sb_abs, ws_sb_abs, fs_sb_abs, cs_sb_abs, ts_sb_abs = self.compute_gs_sb_abs(torch.ones_like(self.seed_indices, dtype=torch.bool))

        v_sb_abs = create_cholesky(self.diags_act(diags_sb_abs), self.l_triangs_act(l_triangs_sb_abs))

        if invert is True:
            v_sb_abs_inv = torch.inverse(v_sb_abs)
        else:
            v_sb_abs_inv = v_sb_abs

        return m_sb_abs, v_sb_abs, v_sb_abs_inv, self.ws_act(ws_sb_abs), self.fs_act(fs_sb_abs), cs_sb_abs, ts_sb_abs

    def get_gs_total(self, invert=True):

        if self.n_gs_sb == 0:
            return self.get_gs(invert)
        else:
            m, v, v_inv, ws, fs, cs, ts = self.get_gs(invert)
            m_sb, v_sb, v_inv_sb, ws_sb, fs_sb, cs_sb, ts_sb = self.get_gs_sb(invert)

            total_m = torch.cat([m, m_sb], dim=0)
            total_v = torch.cat([v, v_sb], dim=0)
            total_v_inv = torch.cat([v_inv, v_inv_sb], dim=0)
            total_ws = torch.cat([ws, ws_sb], dim=0)
            total_fs = torch.cat([fs, fs_sb], dim=0)
            total_cs = torch.cat([cs, cs_sb], dim=0)
            total_ts = torch.cat([ts, ts_sb], dim=0)

            return total_m, total_v, total_v_inv, total_ws, total_fs, total_cs, total_ts

    def cull(self, q, total_m, total_v):

        q_projections, _ = project_vectors_gaussians(vecs=q, projection_vecs=self.projection_vecs, n_hashes=self.n_projection_vecs)

        m_projections, m_projections_range = project_vectors_gaussians(vecs=total_m, projection_vecs=self.projection_vecs, cov=total_v, n_hashes=self.n_projection_vecs)

        mask = EvaluatorLSH.cull(q_projections.contiguous(), m_projections.contiguous(), m_projections_range.contiguous())

        return mask

    def construct_list_of_attributes(self):
        l = []

        for i in range(self.m.shape[1]):
            l.append('m_{}'.format(i))

        for i in range(self.diags.shape[1]):
            for j in range(self.diags.shape[1]):
                l.append('cov_'+str(i)+'_'+str(j))

        for i in range(self.fs.shape[1]):
            l.append('f_{}'.format(i))

        l.append('opacity')

        return l

    def save_ply(self, path):

        m, v, v_inv, ws, fs, cs, ts = self.get_gs_total(invert=False)

        m = m.detach().cpu().numpy()
        v = v.detach().flatten(start_dim=1).cpu().numpy()
        fs = fs.detach().cpu().numpy()
        ws = ws.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(m.shape[0], dtype=dtype_full)
        attributes = np.concatenate((m, v, fs, ws), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def forward(self, x, view=0):

        q = x[:, :self.gs_dim]

        total_m, total_v, total_v_inv, total_ws, total_fs, total_cs, total_ts = self.get_gs_total()

        q_projections, _ = project_vectors_gaussians(vecs=q, projection_vecs=self.projection_vecs, n_hashes=self.n_projection_vecs)

        m_projections, m_projections_range = project_vectors_gaussians(vecs=total_m, projection_vecs=self.projection_vecs, cov=total_v, n_hashes=self.n_projection_vecs)

        if view == 0:
            out, out_evals, out_tile_evals, out_ids = EvaluatorLSH.apply(q.contiguous(), q_projections.contiguous(), total_m.contiguous(), m_projections.contiguous(), m_projections_range.contiguous(), total_v_inv.contiguous(), (total_ws*total_fs).contiguous(), 20000)
        elif view == 1:
            out, out_evals, out_tile_evals, out_ids = EvaluatorLSH.apply(q.contiguous(), q_projections.contiguous(), total_m.contiguous(), m_projections.contiguous(), m_projections_range.contiguous(), total_v_inv.contiguous(), (total_ws*total_cs).contiguous(), 20000)

        return out

