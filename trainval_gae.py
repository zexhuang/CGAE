import yaml
import pandas as pd

from torch_geometric.loader import DataLoader
from data.dataset import PolygonDataset
from train.trainer import Trainer
from model.nn import build_model


if __name__ == "__main__": 
    with open('cfg/gae.yaml', 'r') as f:
        cfg = yaml.safe_load(f)    
        
    train_df = pd.read_pickle(cfg['train'])    
    val_df = pd.read_pickle(cfg['val'])
        
    train_set = PolygonDataset(train_df, 
                               ratio=cfg['ratio'])
    train_loader = DataLoader(train_set, 
                              batch_size=cfg['batch'],
                              num_workers=cfg['worker'])
    
    val_set = PolygonDataset(val_df, 
                             ratio=cfg['ratio'])
    val_loader = DataLoader(val_set, 
                            batch_size=cfg['batch'],
                            num_workers=cfg['worker'])
        
    for nn in ['edgc', 'gin', 'gcn']:
        cfg['nn'] = nn
        if cfg['aug']:
            cfg['path'] = f'save/aug/{cfg["nn"]}'
        else:
            cfg['path'] = f'save/no_aug/{cfg["nn"]}'           
        model = build_model(cfg=cfg)
        trainer = Trainer(cfg=cfg) 
        trainer.fit(model, 
                    train_loader=train_loader, 
                    val_loader=val_loader,
                    early_stop=False)  
        
