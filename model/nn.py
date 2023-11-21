import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import PReLU
from torch_geometric.nn import LayerNorm, MLP
from torch_geometric.nn import (GCNConv, GINConv, 
                                EdgeConv, Linear)


class GCNBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        hid_channels,
        out_channels
        ):
        super(GCNBlock, self).__init__()
        self.conv1 = GCNConv(in_channels, 
                             hid_channels)
        self.conv2 = GCNConv(hid_channels, 
                             out_channels)
        self.norm = LayerNorm(hid_channels)
        self.act = PReLU()
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        return self.conv2(x, edge_index)
    
    
class GINBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        hid_channels,
        out_channels):
        super(GINBlock, self).__init__()
        self.conv1 = GINConv(Linear(in_channels, hid_channels), 
                             eps=0.0001, 
                             train_eps=True, 
                             aggr='max')
        self.conv2 = GINConv(Linear(hid_channels, out_channels), 
                             eps=0.0001, 
                             train_eps=True, 
                             aggr='max')
        self.norm = LayerNorm(hid_channels)
        self.act = PReLU()
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        return self.conv2(x, edge_index)
    

class EdgeBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        hid_channels,
        out_channels):
        super(EdgeBlock, self).__init__()
        self.conv1 = EdgeConv(Linear(in_channels*2, 
                                     hid_channels))
        self.conv2 = EdgeConv(Linear(hid_channels*2, 
                                     out_channels))
        self.norm = LayerNorm(hid_channels)
        self.act = PReLU()
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        return self.conv2(x, edge_index)
    

class GAE(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hid_channels: int, 
        conv: nn.Module,
        ):
        super(GAE, self).__init__()   
        self.E = conv(in_channels, 
                      hid_channels, 
                      hid_channels)
        
        self.D = conv(hid_channels, 
                      hid_channels, 
                      in_channels)
    
    def _rec_edge_loss(self, z, pos_edge_index, neg_edge_index):
        loss = nn.BCELoss()
        
        pos_edge = (z[pos_edge_index[0]] 
                    * z[pos_edge_index[1]]).sum(dim=-1).view(-1).sigmoid()
        neg_edge = (z[neg_edge_index[0]] 
                    * z[neg_edge_index[1]]).sum(dim=-1).view(-1).sigmoid()
        
        ones = torch.ones_like(pos_edge)
        zeros = torch.zeros_like(neg_edge)
        
        pos_loss = loss(pos_edge, ones)
        neg_loss = loss(neg_edge, zeros)
        return pos_loss + neg_loss
    
    def _rec_node_loss(self, z, edge_index, x):
        loss = nn.MSELoss()
        z = self.D(z, edge_index)
        return loss(z, x)
    
    def _nt_xent_loss(self, z_i, z_j, t=0.5, eps=1e-6):
        n_z_i = F.normalize(z_i, dim=-1)
        n_z_j = F.normalize(z_j, dim=-1)
        
        z = torch.cat([n_z_i, n_z_j], dim=0)
        n_samples = len(z)

        # Full similarity matrix
        sim_mat = torch.mm(z, z.t().contiguous())
        similarity = torch.exp(sim_mat / t)

        mask = ~torch.eye(n_samples, device=similarity.device).bool()
        neg = similarity.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Positive similarity
        pos = torch.exp(torch.sum(n_z_i * n_z_j, dim=-1) / t)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()
        return loss
    
    def encoder(self, x, edge_index):            
        z = self.E(x, edge_index)
        return z
    
    def decode(self, x, edge_index):
        z = self.encoder(x, edge_index)
        adj = torch.matmul(z, z.t())
        return adj.sigmoid().fill_diagonal_(0), z
    
    
class NUFTAE(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hid_channels: int):
        super(NUFTAE, self).__init__()   
        self.E = MLP([in_channels, 
                      hid_channels, 
                      hid_channels,
                      hid_channels], 
                      dropout=0.25,
                      plain_last=True)
        
        self.D = MLP([hid_channels, 
                      hid_channels, 
                      hid_channels,
                      in_channels], 
                      dropout=0.25,
                      plain_last=True)
        
    def _rec_loss(self, z, batch, x):
        loss = nn.MSELoss()
        z = self.D(z, batch)
        return loss(z, x)
        
    def encoder(self, x, batch):            
        z = self.E(x, batch)
        return z
    
    
import yaml
from pathlib import Path
from typing import Union

def build_model(cfg: Union[str, Path, dict]):
    if isinstance(cfg, str or Path):
        with open(cfg, 'r') as f: 
            cfg = yaml.safe_load(f)    
            
    if cfg['nn'] == 'gcn':
        return GAE(
            in_channels=cfg['in_channels'],
            hid_channels=cfg['hid_channels'],
            conv=GCNBlock
            )
    elif cfg['nn'] == 'gin':
        return GAE(
            in_channels=cfg['in_channels'],
            hid_channels=cfg['hid_channels'],
            conv=GINBlock
            )
    elif cfg['nn'] == 'edgc':
        return GAE(
            in_channels=cfg['in_channels'],
            hid_channels=cfg['hid_channels'],
            conv=EdgeBlock
            )
    elif cfg['nn'] == 'nuft':
        return NUFTAE(in_channels=cfg['in_channels'], 
                      hid_channels=cfg['hid_channels'])
