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
import torch.nn.functional as F
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

class ResBlock(nn.Module):
    def __init__(self, hidden_dim, nn_dropout=0.0, skip=True):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.do1 = nn.Dropout(nn_dropout)
        self.do2 = nn.Dropout(nn_dropout)
        self.act = nn.ReLU()
        self.skip = skip
        self.skip_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x_ = self.do1(self.act(self.l1(x)))
        x_ = self.do2(self.act(self.l2(x_)))
        if self.skip:
            return x_ + self.skip_layer(x)
        else:
            return x_

class NodeEmbeddings(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.node_dim = node_dim
        self.l1 = nn.Linear(node_dim, node_dim)
        self.l2 = nn.Linear(node_dim, node_dim)
        self.res = nn.Linear(node_dim, node_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(-1).expand(-1, -1, self.node_dim)
        res = self.res(x)
        h = self.l2(self.act(self.l1(x)))
        return h + res

class DDCD_NonLinear(nn.Module):
    def __init__(
        self, d, init_coef=1e-8, adj_dropout=0.0,
        time_dim=16, hidden_dims=[16, 16, 16], nn_dropout=0.0, skip=True,
        estimate_variance=False
    ):
        super(DDCD_NonLinear, self).__init__()
        
        self.d = d
        self.init_coef = init_coef
        self.adj_dropout = adj_dropout
        self.time_dim = time_dim
        self.hidden_dims = hidden_dims
        self.nn_dropout = nn_dropout
        self.node_dim = hidden_dims[0]
        self.estimate_variance = estimate_variance

        adj_A = torch.randn(2, d, d) * init_coef
        self.adj_A = nn.Parameter(adj_A, requires_grad=True)
        if estimate_variance:
            adj_A_std = torch.zeros(2, d, d) + 1e-8
            self.adj_A_std = nn.Parameter(adj_A_std, requires_grad=True)

        self.x_emb = NodeEmbeddings(self.node_dim)
        self.x_to_y = nn.Sequential(
            ResBlock(self.node_dim, nn_dropout),
            nn.Linear(self.node_dim, 1)
        )
        self.yw_emb = NodeEmbeddings(self.node_dim)
        self.yw_to_x = nn.Sequential(
            ResBlock(self.node_dim, nn_dropout, skip),
            nn.Linear(self.node_dim, 1)
        )

        sample_size = min(d * d, 10000)
        self.sampled_adj_row_nonparam = nn.Parameter(
            torch.randint(0, d, size=(sample_size,)), 
            requires_grad=False)
        self.sampled_adj_col_nonparam = nn.Parameter(
            torch.randint(0, d, size=(sample_size,)),
            requires_grad=False)
        self.eye_nonparam = nn.Parameter(torch.eye(d), requires_grad=False)

    def forward(self, y):
        w = self.get_adj_with_dropout()
        return y - y@w

    def transform_x_to_y(self, x):
        h_x = self.x_emb(x)
        return self.x_to_y(h_x).squeeze(-1)

    def transform_yw_to_x(self, yw):
        h_yw = self.yw_emb(yw)
        return self.yw_to_x(h_yw).squeeze(-1)

    def get_adj_(self):
        return (self.adj_A[0, :, :]-self.adj_A[1, :, :]) * (1-self.eye_nonparam)
    
    def get_adj(self):
        adj = self.get_adj_().detach().cpu().numpy() 
        return adj / (np.abs(adj).max())
    
    def get_adj_with_dropout(self):
        if self.estimate_variance:
        # reparamerization
            adj_A = self.adj_A + self.adj_A_std * torch.randn_like(self.adj_A)
        else:
            adj_A = self.adj_A
        adj_A = (adj_A[0, :, :] - adj_A[1, :, :]) * (1-self.eye_nonparam)
        A_dropout = (torch.rand_like(adj_A)>self.adj_dropout).float()
        A_dropout /= (1-self.adj_dropout)
        return adj_A * A_dropout   

    def get_i_minus_w_inv(self):
        return torch.inverse(self.eye_nonparam - self.get_adj_())

    @torch.no_grad()
    def get_sampled_adj_(self):
        return self.get_adj_()[
            self.sampled_adj_row_nonparam, self.sampled_adj_col_nonparam
        ].detach()


class DDCD_NonLinear_Trainer:
    def __init__(
        self, X, 
        noise_schedule='linear', T=5000, start_noise=0.0001, end_noise=0.002,
        time_dim=16, hidden_dims=[16, 16, 16], nn_dropout=0.1, skip=True,
        init_coef=0.0, lr=1e-2, batch_size=128, n_steps=10000, device='cuda', 
        adj_l1_coef=0.1, adj_l2_coef=0.01, adj_dropout=0.00,
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

        X_ = X - np.mean(X, axis=0, keepdims=True)

        x_tensor = torch.tensor(X_, dtype=float_type)
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
        self.model = DDCD_NonLinear(
            d=d, init_coef=init_coef, adj_dropout=adj_dropout,
            time_dim=time_dim, hidden_dims=hidden_dims, nn_dropout=nn_dropout,
            estimate_variance=estimate_variance, skip=skip
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
            amsgrad=True
        )
        def lr_lambda(current_step):
            if current_step < 1000:
                return 1
            else:
                return 1
                
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

        self.model.to(device)
        self.total_time_cost=0
        self.lr = lr
        self.adjs = []
        self.model_name='DDCD_NonLinear'

    def update_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    @torch.no_grad()
    def forward_pass(self, y_0):
        """
        Forward diffusion process
        
        Args:
            x_0 (torch.FloatTensor): Torch tensor for expression data. Rows are 
            cells and columns are genes
        """
        t = torch.randint(
            0, self.hp['T'], (y_0.shape[0],), 
            device=self.device
        ).long()
        noise = torch.randn_like(y_0)
        mean_coef = self.mean_schedule.gather(dim=-1, index=t)
        std_coef = self.std_schedule.gather(dim=-1, index=t)
        y_t = mean_coef.unsqueeze(-1) * y_0 + std_coef.unsqueeze(-1) * noise
        return t, y_t, noise, mean_coef, std_coef

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
                epoch_loss1 = []
                epoch_loss2 = []
                epoch_hA = []
                epoch_l1 = []
                epoch_l2 = []
                for step, batch in enumerate(self.train_dataloader):
                    x = batch[0]
                    x = x.to(self.device)
                    self.opt.zero_grad()
                    y_0 = self.model.transform_x_to_y(x)
                    t, y_t, noise, mean_coef, std_coef = self.forward_pass(y_0)

                    z = self.model(y_t)
                    adj_m = self.model.get_adj_()
                    adj_2m = self.model.adj_A
                    yw = y_0 @ adj_m
                    x_pred = self.model.transform_yw_to_x(yw)
                    expected_noise = (noise - noise @ adj_m) * std_coef.view(-1, 1)
                    
                    loss_1 = F.mse_loss(expected_noise, z, reduction='mean')
                    loss_2 = F.mse_loss(x, x_pred, reduction='mean')
                    loss = loss_1 + loss_2
        
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
                    epoch_loss1.append(loss_1.item())
                    epoch_loss2.append(loss_2.item())
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
                train_loss1 = np.mean(epoch_loss1)
                train_loss2 = np.mean(epoch_loss2)
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
                    'train_loss': train_loss, 
                    'train_loss1': train_loss1,
                    'train_loss2': train_loss2,
                    'train_hA': train_hA, 
                    'adj_change': adj_diff, 
                    'train_l1': train_l1, 'train_l2': train_l2}
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