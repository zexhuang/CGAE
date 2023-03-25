import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="dark", 
              palette="pastel")
import numpy as np


def draw_graph(pos: np.ndarray, 
               edge: np.ndarray, 
               mask=None, 
               ax=None):      
    x = [pos[:,0][edge[:,0]], pos[:,0][edge[:,1]]] # x_src, x_tgt
    y = [pos[:,1][edge[:,0]], pos[:,1][edge[:,1]]] # y_src, y_tgt
    mask = np.ones(pos.shape[0]) if mask is None else mask
    # Subplots
    if ax == None: ax = plt.subplot(1, 1, 1)  
    # Edges
    ax.plot(x, y, 'b-')         
    # Nodes
    sns.scatterplot(x=pos[:,0], 
                    y=pos[:,1], 
                    hue=mask, 
                    palette="YlOrBr", 
                    markers='o', 
                    legend=False, 
                    s=60, ax=ax)
    return ax