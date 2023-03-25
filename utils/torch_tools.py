import numpy as np
import torch

# Code adapoted from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation monitor doesn't improve after a given patience."""
    def __init__(self, path, best_score=None, patience=10, delta=0.0, verbose=False, trace_func=print):
        """
        Args:
            path (str): Path for the checkpoint to be saved to.
            best_score (flaot or none): Value of metric of the best model.
                            Default: None
            patience (int): How long to wait after last time validation monitor improved.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation monitor improvement. 
                            Default: False
            trace_func (function): trace print function.
                            Default: print            
        """
        self.path = path
        self.best_score = best_score
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.trace_func = trace_func
        
        self.counter = 0
        self.early_stop = False

    def __call__(self, monitor, cm, model, optimizer, epoch):
        score = monitor # loss
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'monitor': monitor,
            'confusion_matrix': cm
        }
    
        if self.best_score is None:
            self.save_checkpoint(checkpoint, score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'Validation monitor does not imporove ({self.best_score} --> {score}). \n EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(checkpoint, score)
            self.counter = 0

    def save_checkpoint(self, checkpoint, score):
        '''Saves model when validation monitor decrease/increase.'''
        if self.verbose:
            if self.best_score is None:
                self.best_score = np.inf

            self.trace_func(f'Validation monitor decrease/increase ({self.best_score:.6f} --> {score:.6f}). \n Saving model ...')
            
        torch.save(checkpoint, self.path)
        self.best_score = score