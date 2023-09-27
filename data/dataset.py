import torch 
import pathlib
import pandas as pd

from typing import Union
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import NormalizeScale
from torch_geometric.utils import (negative_sampling, 
                                   add_random_edge, 
                                   dropout_node,
                                   coalesce,
                                   mask_feature)
    

class PolygonDataset(Dataset):
    def __init__(self, 
                 dataset: Union[str, pathlib.Path, pd.DataFrame], 
                 ratio: float = 0.2,
                 transform=NormalizeScale()):
        super().__init__()   
        # Dataframe
        if isinstance(dataset, str or pathlib.Path):
            df = pd.read_pickle(dataset) 
        elif isinstance(dataset, pd.DataFrame):
            df = dataset
        # Graph feature                
        self.pos = list(df['pos'])
        self.edge = list(df['contour'])
        self.ratio = ratio
        self.transform = transform
            
    def len(self):
        """Return the number of samples in the custom dataset."""
        return len(self.pos)

    def get(self, idx):
        """Loda and return a data sample from the custom dataset at given index."""
        data = Data() 
        data.pos = torch.from_numpy(self.pos[idx].T).float()
        data.num_nodes = data.pos.size(0)
        data.edge_index = torch.from_numpy(self.edge[idx]).long()
        neg_edge_index = negative_sampling(edge_index=data.edge_index,
                                           num_nodes=data.num_nodes,
                                           num_neg_samples=data.edge_index.size(1), 
                                           method='dense')     
        data.neg_edge_index = coalesce(edge_index=neg_edge_index, 
                                       num_nodes=data.num_nodes)
        # Edge augmentation
        aug_edge_index, _ = add_random_edge(edge_index=data.edge_index, 
                                            p=self.ratio, 
                                            num_nodes=data.num_nodes)
        data.aug_edge_index = coalesce(edge_index=aug_edge_index, 
                                       num_nodes=data.num_nodes)
        # Node reduction        
        data.sub_edge_index, \
        data.sub_edge_mask, \
        data.sub_node_mask = dropout_node(edge_index=data.edge_index, 
                                          p=self.ratio, 
                                          num_nodes=data.num_nodes)
        if self.transform:
            data = self.transform(data)
        return data