import yaml
import pandas as pd

from torch_geometric.loader import DataLoader
from data.dataset import NUFTDataset
from train.nuft_trainer import Trainer
from model.nn import build_model


if __name__ == "__main__": 
    with open('cfg/nuft.yaml', 'r') as f:
        cfg = yaml.safe_load(f)    
        
    train_df = pd.read_pickle(cfg['train'])    
    val_df = pd.read_pickle(cfg['val'])
        
    train_set = NUFTDataset(train_df, cfg['res'], cfg['norm']).to_data_list()
    val_set = NUFTDataset(val_df, cfg['res'], cfg['norm']).to_data_list()
    
    train_loader = DataLoader(train_set, 
                              batch_size=cfg['batch'],
                              num_workers=cfg['worker'],
                              shuffle=True)
    val_loader = DataLoader(val_set, 
                            batch_size=cfg['batch'],
                            num_workers=cfg['worker'],
                            shuffle=False)
    for nn in ['nuft']:
        cfg['nn'] = nn
        cfg['path'] = f'save/{cfg["nn"]}'          
        model = build_model(cfg=cfg)
        trainer = Trainer(cfg=cfg) 
        trainer.fit(model, 
                    train_loader=train_loader, 
                    val_loader=val_loader,
                    early_stop=False) 