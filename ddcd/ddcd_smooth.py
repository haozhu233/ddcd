import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
from datetime import datetime
import warnings
import math
from .logger import LightLogger
from .utils import linear_beta_schedule, power_beta_schedule, cosine_beta_schedule, dag_h, dag_h_khop, count_accuracy

import torch
import numpy as np
from torch import nn
from torch.distributions import Normal
import math
from typing import List

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.theta) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim, nn_dropout=0.0):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_dim)
        self.l1 = nn.Linear(in_dim, out_dim)
        self.l2 = nn.Linear(out_dim, out_dim)
        self.do1 = nn.Dropout(nn_dropout)
        self.act = nn.Tanh()
        
    def forward(self, x, t):
        h = self.do1(self.act(self.l1(x)))
        time_emb = self.act(self.time_mlp(t)).unsqueeze(1)
        h = h + time_emb 
        h = self.act(self.l2(h))
        return h
    
class NodeEmbeddings(nn.Module):
    def __init__(self, d, node_dim):
        super().__init__()
        node_emb = torch.randn(d, node_dim-1)
        self.node_emb = nn.Parameter(node_emb, requires_grad=True)

    def forward(self, x):
        n = x.shape[0]
        batch_node_emb = self.node_emb.unsqueeze(0).repeat(n, 1, 1)
        batch_node_emb = torch.concat([x.unsqueeze(-1), batch_node_emb], dim=-1)
        return batch_node_emb


# class NodeEmbeddings(nn.Module):
#     def __init__(self, node_dim):
#         super().__init__()
#         self.node_dim = node_dim
#         self.l1 = nn.Linear(node_dim, node_dim)
#         self.l2 = nn.Linear(node_dim, node_dim)
#         self.res = nn.Linear(node_dim, node_dim)
#         self.act = nn.ReLU()

#     def forward(self, x):
#         x = x.unsqueeze(-1).expand(-1, -1, self.node_dim)
#         res = self.res(x)
#         h = self.l2(self.act(self.l1(x)))
#         return h + res

class DDCD_Smooth(nn.Module):
    def __init__(
        self, d, init_coef=1e-8, adj_dropout=0.0,
        time_dim=16, hidden_dims=[16, 16, 16], nn_dropout=0.0,
        estimate_variance=False
    ):
        super(DDCD_Smooth, self).__init__()
        
        self.d = d
        self.init_coef = init_coef
        self.adj_dropout = adj_dropout
        self.time_dim = time_dim
        self.hidden_dims = hidden_dims
        self.nn_dropout = nn_dropout
        self.node_dim = hidden_dims[0]
        self.estimate_variance = estimate_variance

        # adj_A = torch.ones(d, d) * init_coef
        adj_A = torch.ones(d, d) /(d-1) * init_coef
        self.adj_A = nn.Parameter(adj_A, requires_grad =True, )
        
        if estimate_variance:
            adj_A_std = torch.zeros(d, d) + 1e-8
            self.adj_A_std = nn.Parameter(adj_A_std, requires_grad=True)
        
        sample_size = min(d * d, 10000)
        self.sampled_adj_row_nonparam = nn.Parameter(
            torch.randint(0, d, size=(sample_size,)), 
            requires_grad=False)
        self.sampled_adj_col_nonparam = nn.Parameter(
            torch.randint(0, d, size=(sample_size,)),
            requires_grad=False)

        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Tanh()
        )
        
        self.node_emb = nn.Sequential(
            NodeEmbeddings(self.d, self.node_dim),
            # NodeEmbeddings(self.node_dim),
            nn.Tanh()
        )
        
        self.blocks = nn.ModuleList([
            Block(
                hidden_dims[i], hidden_dims[i+1]-1, time_dim, nn_dropout
            ) for i in range(len(hidden_dims) - 1)
        ])
        
        self.final = nn.Sequential(
            nn.Linear(hidden_dims[-1]-1, hidden_dims[-1]-1),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1]-1, 1)
        )
        self.identity = nn.Identity()

    def forward(self, x, t):
        w = self.get_adj_with_dropout()

        h_x = self.emb_x(x, t)
        h_xw = torch.einsum('ogd,gh->ohd', h_x, w)
        h_z = h_x - h_xw
        z = self.final(h_z).squeeze(-1)
        return z

    def emb_x(self, x, t):
        original = x.unsqueeze(-1)
        h_time = self.time_emb(t)
        h_x = self.node_emb(x)
        # h_0 = self.identity(h_x)
        for i, block in enumerate(self.blocks):
            if i != 0:
                h_x = torch.concat([h_x, original], dim=-1)
            h_x = block(h_x, h_time)
            # h_x = h_x + h_0
        return h_x
    
    # def emb_x(self, x, t):
    #     # original = x.unsqueeze(-1)
    #     h_time = self.time_emb(t)
    #     h_x = self.node_emb(x)
    #     h_0 = self.identity(h_x)
    #     for i, block in enumerate(self.blocks):
    #         h_x = block(h_x, h_time)
    #         h_x = h_x + h_0
    #     return h_x

    def get_adj_(self):
        return self.adj_A
    
    def get_adj(self):
        adj = self.get_adj_().detach().cpu().numpy() 
        return adj / (np.abs(adj).max())
    
    def get_adj_with_dropout(self):
        if self.estimate_variance:
        # reparamerization
            adj_A = self.adj_A + self.adj_A_std * torch.randn_like(self.adj_A)
        else:
            adj_A = self.adj_A
        adj_A = adj_A
        A_dropout = (torch.rand_like(adj_A)>self.adj_dropout).float()
        A_dropout /= (1-self.adj_dropout)
        return adj_A * A_dropout       

    @torch.no_grad()
    def get_sampled_adj_(self):
        return self.get_adj_()[
            self.sampled_adj_row_nonparam, self.sampled_adj_col_nonparam
        ].detach()


class DDCD_Smooth_Trainer:
    def __init__(
        self, X, 
        noise_schedule='linear', T=5000, start_noise=0.0001, end_noise=0.001,
        time_dim=16, hidden_dims=[16, 16, 16], nn_dropout=0.1,
        init_coef=1e-8, lr=0.001, batch_size=128, n_steps=1000, device='cuda', 
        adj_l1_coef=0.01, adj_l2_coef=0.01, adj_dropout=0.00,
        dag_control_coef=1, dag_steps = 1,
        dag_scaling_factor=3.0,
        estimate_variance=False, pbar=True
    ):
        hp = locals()
        del hp['X']
        self.hp = hp
        
        if device == 'mps':
            raise Exception("We noticed unreliable training behavior on", 
                            "Apple's silicon. Consider using other devices.")
        elif device.startswith('cuda'):
            if not torch.cuda.is_available():
                print(
                    "You specified cuda as your computing device but apprently", 
                    "it's not available. Setting device to cpu for now. ")
                device = 'cpu'
        self.device = device
        self.hp['device'] = device
        float_type = torch.float32

        # Logger ---------------------------------------------------------------
        self.logger = LightLogger()
        self.note_id = self.logger.start()
        
        # Define diffusion schedule
        if noise_schedule == 'linear':
            self.betas = linear_beta_schedule(T, start_noise, end_noise)
        elif noise_schedule == 'power':
            self.betas = power_beta_schedule(T, start_noise, end_noise, power=2)
        elif noise_schedule == 'cosine':
            self.betas = cosine_beta_schedule(T, s=0.008)
            
        self.alphas = 1. - self.betas
        alpha_bars = torch.cumprod(self.alphas, axis=0)
        self.mean_schedule = torch.sqrt(alpha_bars).to(device)
        self.std_schedule = torch.sqrt(1. - alpha_bars).to(device)
    
        # Prepare Data ---------------------------------------------------------
        if (X.sum(0) == 0).sum() > 0:
            warnings.warn(
                "Some columns in the X contains all zero values, "
                "which often causes trouble in inference. Please consider "
                "removing these columns before continuing. "
            )

        n, d = X.shape
        self.n = n
        self.d = d

        x_tensor = torch.tensor(X, dtype=float_type)
        self.x_std_tensor = torch.std(x_tensor, dim=0).to(self.device)
    
        ## Setup dataset and dataloader
        self.train_dataset = torch.utils.data.TensorDataset(x_tensor)
        # Implement bootstrap for train sampler
        train_sampler = torch.utils.data.RandomSampler(
            self.train_dataset, replacement=True, num_samples=batch_size)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            sampler=train_sampler,
            batch_size = batch_size, 
            drop_last=True)
    
        # Setup Model ----------------------------------------------------------
        self.model = DDCD_Smooth(
            d=d, init_coef=init_coef, adj_dropout=adj_dropout,
            time_dim=time_dim, hidden_dims=hidden_dims, nn_dropout=nn_dropout,
            estimate_variance=estimate_variance
        )

        # Setup optimizer ------------------------------------------------------
        model_params = []
        for name, param in self.model.named_parameters():
            if not name.endswith('_nonparam'):
                model_params.append(param)
        self.opt = torch.optim.Adam(
            model_params, 
            lr=lr, 
            weight_decay=0.0,
            betas=[0.9, 0.999],
            # decoupled_weight_decay=True
            # amsgrad=True
        )
        def lr_lambda(current_step):
            if current_step < 500:
                return 1e-2
            else:
                return 1
                
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

        self.model.to(device)
        self.total_time_cost=0
        self.lr = lr
        self.adjs = []
        self.model_name='DDCD'

    @torch.no_grad()
    def forward_pass(self, x_0):
        """
        Forward diffusion process
        
        Args:
            x_0 (torch.FloatTensor): Torch tensor for expression data. Rows are 
            cells and columns are genes
        """
        x_0 = x_0.to(self.device)
        t = torch.randint(
            0, self.hp['T'], (x_0.shape[0],), 
            device=self.device
        ).long()
        # noise = self.x_std_tensor.unsqueeze(0) * torch.randn_like(x_0)
        noise = torch.randn_like(x_0)
        mean_coef = self.mean_schedule.gather(dim=-1, index=t)
        std_coef = self.std_schedule.gather(dim=-1, index=t)
        x_t = mean_coef.unsqueeze(-1) * x_0 + std_coef.unsqueeze(-1) * noise
        return t, x_t, noise, std_coef

    def train(self, n_steps=None):
        """
        Train the initialized model for a number of steps. 

        Args:
            n_steps (int): Number of steps to train. If not provided, it will 
                train the model by the n_steps sepcified in class 
                initialization. Please read our paper to see how to identify
                the converge point. 
        """
        start_time = datetime.now()
        
        if n_steps is None:
            n_steps = self.hp['n_steps']
            
        sampled_adj = self.model.get_sampled_adj_()
        train_hA = None
        start_step = len(self.logger.mem[self.logger.current_log]['log'])

        with tqdm(range(n_steps), disable=not self.hp['pbar']) as pbar:
            for epoch_ in pbar: 
                epoch = epoch_ + start_step
                if epoch < 0.4 * n_steps:
                    current_khop = 3
                elif epoch < 0.9 * n_steps:
                    current_khop = 10
                else:
                    current_khop = self.d
                    
                epoch_loss = []
                epoch_hA = []
                epoch_l1 = []
                epoch_l2 = []
                for step, batch in enumerate(self.train_dataloader):
                    x_0 = batch[0]
                    x_0 = x_0.to(self.device)

                    
                    t, x_noisy, noise, std_coef = self.forward_pass(x_0)

                    self.opt.zero_grad()
                    z = self.model(x_noisy, t)
                    adj_m = self.model.get_adj_()
                    adj_2m = self.model.adj_A
        
                    expected_noise = (noise - noise @ adj_m) * std_coef.view(-1, 1)
                    
                    loss = F.mse_loss(expected_noise, z, reduction='mean')
        
                    loss_l1 = adj_2m.abs().mean() * self.hp['adj_l1_coef']
                    loss_l2 = (adj_2m * adj_2m).mean() * self.hp['adj_l2_coef']
        
                    loss = loss + loss_l1 + loss_l2
                    
                    if epoch % self.hp['dag_steps'] == self.hp['dag_steps'] - 1:
                        h_A = dag_h_khop(
                            adj_m, current_khop, 
                            self.hp['dag_scaling_factor']
                            )
                        loss = loss + self.hp['dag_control_coef'] * h_A * self.hp['dag_steps'] * np.sqrt(epoch)
                    loss.backward()
                    epoch_loss.append(loss.item())
                    epoch_l1.append(loss_l1.item())
                    epoch_l2.append(loss_l2.item())
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
                    self.opt.step()
                    self.scheduler.step()
                    total_grad = 0.0
                    
                    for name, param in self.model.named_parameters():
                        if name.startswith('adj_A'):
                            total_grad += param.grad.abs().sum().item() 
                    if epoch % self.hp['dag_steps'] == self.hp['dag_steps'] - 1:
                        with torch.no_grad():
                            adj_m = self.model.get_adj_()
                            h_A = dag_h_khop(
                                adj_m, current_khop, 
                                self.hp['dag_scaling_factor']
                            ).item()
                            epoch_hA.append(h_A)
                train_loss = np.mean(epoch_loss)
                if len(epoch_hA) != 0:
                    train_hA = np.mean(epoch_hA)
                train_l1 = np.mean(epoch_l1)
                train_l2 = np.mean(epoch_l2)
                sampled_adj_new = self.model.get_sampled_adj_()
                adj_diff = (
                    sampled_adj_new - sampled_adj
                    ).mean().item()*(self.d-1)
                sampled_adj = sampled_adj_new
                pbar.set_description(
                    f'Training loss: {train_loss:.4f}, train_hA: {train_hA:.4f}, Change on Adj: {adj_diff:.4f}, grad: {total_grad:.4f}')
                epoch_log = {
                    'train_loss': train_loss, 'train_hA': train_hA, 
                    'adj_change': adj_diff, 'train_l1': train_l1, 'train_l2': train_l2}
                self.logger.log(epoch_log)
                # if epoch % 50 == 0:
                #     self.adjs.append(self.model.get_adj())
        self.total_time_cost += int(
            (datetime.now() - start_time).total_seconds())
        return None

    def get_adj(self):
        """
        Obtain the adjacency matrix. The values in this adjacency matix has 
        been scaled using regulatory norm. You may expect strong links to go
        beyond 5 or 10 in most cases. 
        """
        return self.model.get_adj()