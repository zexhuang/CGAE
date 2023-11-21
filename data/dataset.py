import torch 
import pathlib
import pandas as pd

from tqdm import tqdm
from typing import Union
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import NormalizeScale
from torch_geometric.utils import (negative_sampling, 
                                   add_random_edge, 
                                   dropout_node,
                                   coalesce)
    

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
    
    
import numpy as np

from typing import Any
from shapely import wkt
from DDSL.experiments.exp2_mnist.loader import poly2ve
from DDSL.ddsl.ddsl import *

class NUFTDataset(object):
    def __init__(self, 
                 dataset: Union[str, pathlib.Path, pd.DataFrame], 
                 res: int = 16,
                 embed_norm: Any = 'l2'):
        super().__init__()   
        # Dataframe
        if isinstance(dataset, str or pathlib.Path):
            df = pd.read_pickle(dataset) 
        elif isinstance(dataset, pd.DataFrame):
            df = dataset
    
        self.res = res
        self.embed_norm = embed_norm
        self.geoms = df.geom.tolist()
        
    def to_data_list(self):
        data_list = []
        
        for id, geom in enumerate(tqdm(self.geoms)):
            P = wkt.loads(geom)
            V, E = poly2ve(P)
            # Rescale to (-1, -1)
            V = V - V.mean(axis=-2, keepdims=True)
            scale = (1 / np.absolute(V).max()) * 0.999999
            V *= scale
            # Random Translation
            V += 1e-6*np.random.rand(*V.shape)
            
            V = torch.tensor(V, dtype=torch.float64, requires_grad=False)
            E = torch.LongTensor(E)
            D = torch.ones(E.shape[0], 1, dtype=torch.float64)
            
            ddsl_spec = DDSL_spec((self.res, self.res), (1,1), 2)
            F = ddsl_spec(V,E,D)
            poly_nuft_embeds = F.flatten().to(V.dtype)
            poly_nuft_embeds[poly_nuft_embeds.isnan()] = 0
            
            if self.embed_norm == None or self.embed_norm == "F":
                poly_nuft_embeds_ = poly_nuft_embeds
            elif self.embed_norm == "l2":
                poly_nuft_embeds_norm = torch.norm(poly_nuft_embeds, p=2, dim=-1, keepdim=True)
                poly_nuft_embeds_ = torch.div(poly_nuft_embeds, poly_nuft_embeds_norm)
                
            data = Data() 
            data.x = poly_nuft_embeds_.view(1, -1).float()
            data_list.append(data)
            
        return data_list