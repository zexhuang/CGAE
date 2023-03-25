import yaml
import logging
import torch

from pathlib import Path
from typing import Union, Optional, Callable
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torchinfo import summary
from utils.torch_tools import EarlyStopping


class Trainer:
    def __init__(self, cfg:Union[str, Path, dict]):
        if isinstance(cfg, str or Path):
            with open(cfg, 'r') as f: self.cfg = yaml.safe_load(f)    
        elif isinstance(cfg, dict): self.cfg = cfg
        self.epoch = self.cfg['epoch']
        self.path = self.cfg['path']
        self.patience = self.cfg['patience']
        self.lr = self.cfg['lr']
        self.w_decay = self.cfg['w_decay']
        self.device = self.cfg['device']
        self.early_stop = EarlyStopping(path=self.path, 
                                        patience=self.patience)
         
    def fit(self, 
            model: Union[torch.nn.Module, MessagePassing],  
            criterion: Optional[Callable]=None,  
            optimizer: Optional[torch.optim.Optimizer]=None, 
            train_loader: Optional[DataLoader]=None,
            val_loader: Optional[DataLoader]=None,
            ckpt: Union[str, Path, None]=None,
            save_period: int=20,
            early_stop: bool=False):
        summary(model)
        model.to(self.device)  
        model.load_state_dict(self._load_ckpt(ckpt)['params']) if ckpt else model          
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.w_decay) 

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                            T_0=self.epoch, 
                                                                            T_mult=1,
                                                                            eta_min=1e-6,
                                                                            verbose=True)
        
        self.writer = SummaryWriter(log_dir=f'{self.path}/runs')
        
        for ep in range(1, self.epoch+1):
            # Train loop
            epoch_t_loss = 0.0
            num_t_batch = 0
            model.train()
            for data in train_loader:
                data.to(self.device)
                optimizer.zero_grad()       
                if self.cfg['aug']:                                        
                    z = model.encoder(data.pos, data.edge_index)                      
                    z_mean = global_mean_pool(z, data.batch)         
                    # Sub graph
                    sub_z = model.encoder(data.pos, data.sub_edge_index) 
                    sub_z_mean = global_mean_pool(sub_z, data.batch)   
                    # Aug edge                    
                    aug_z = model.encoder(data.pos, data.aug_edge_index)     
                    aug_z_mean = global_mean_pool(aug_z, data.batch)         
                    # Contrastive loss
                    c_sub_loss = model._nt_xent_loss(z_mean, sub_z_mean) 
                    c_aug_loss = model._nt_xent_loss(z_mean, aug_z_mean) 
                    # Recon loss
                    e_loss = model._rec_edge_loss(aug_z, data.edge_index, data.neg_edge_index)    
                    n_loss = model._rec_node_loss(sub_z, data.edge_index, data.pos) 
                    loss = c_aug_loss + c_sub_loss + e_loss + n_loss
                else:
                    z = model.encoder(data.pos, data.edge_index)     
                    # Recon loss
                    e_loss = model._rec_edge_loss(z, data.edge_index, data.neg_edge_index)       
                    n_loss = model._rec_node_loss(z, data.edge_index, data.pos)         
                    loss = n_loss + e_loss
                loss.backward()                 # Backward pass 
                optimizer.step()                # Update parameters  
                num_t_batch += 1
                epoch_t_loss += loss.detach().cpu().item() 
            # Eval loop
            model.eval()
            epoch_e_loss = 0.0
            num_e_batch = 0
            for data in val_loader:
                data.to(self.device)
                # Encoder
                z = model.encoder(data.pos, data.edge_index)
                # Recon loss
                e_loss = model._rec_edge_loss(z, data.edge_index, data.neg_edge_index)  
                n_loss = model._rec_node_loss(z, data.edge_index, data.pos)
                loss = n_loss + e_loss
                
                num_e_batch += 1
                epoch_e_loss += loss.detach().cpu().item() 
            # Tensorboard writter
            self.writer.add_scalar('Loss_AE/train', 
                        epoch_t_loss/num_t_batch, 
                        ep)
            self.writer.add_scalar('Loss_AE/val', 
                        epoch_e_loss/num_e_batch, 
                        ep)   
            self.writer.add_scalar('LRate/train', 
                                   lr_scheduler.get_last_lr()[0], 
                                   ep)
            lr_scheduler.step()

            if ep % save_period == 0: 
                self._save_ckpt(model, ckpt_name=f'epoch{ep}')
            
    def load_ckpt(self, 
                  model: Union[torch.nn.Module, MessagePassing], 
                  ckpt: Union[str, Path, None]=None):
        model.load_state_dict(self._load_ckpt(ckpt)['params']) if ckpt else model  
        model.to(self.device)
        model.eval()
        return model
    
    def _save_ckpt(self, model, ckpt_name):
        path = Path(self.path) / 'ckpt'
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({'params':model.state_dict()}, path.joinpath(ckpt_name))
        logging.info('model ckpt is saved.')
    
    def _load_ckpt(self, ckpt_name):
        path = Path(self.path) / 'ckpt'
        return torch.load(path.joinpath(ckpt_name)) # {'params': Tensor}