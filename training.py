import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch import nn
import sklearn
from IPython.display import clear_output
from tqdm.notebook import tqdm as tqdm_notebook
import seaborn as sns
import evaluation
from IPython.display import display
import plotly.express as px
import copy
import os
import sys
import pickle
import random
from IPython.display import clear_output

from .evaluation import Report


device = "cuda" if torch.cuda.is_available() else "cpu"

def set_device(new_device):
    torch.Tensor([]).to(new_device)
    global device 
    device = new_device
    return 


class SLP(nn.Module):
    def __init__(self,input_size = 512, output_size = 9):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, output_size)
        )
        
    def forward(self, x):
        fp = self.layers(x.float())
        return fp
    
class Trainer(Report) :
    def __init__(self, model, loss_fn, optimizer_fn, lr, data_loaders, device = device):
        super().__init__()
        self._model = model
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self._lr = lr
        self.data_loaders = data_loaders
        self.device = device
        self.epochs_trained_on = 0
        self.classification_accuracy = []
        self.validation_losses = [np.inf]
        self.epoch_losses = []
        self.set_optimizer()
    
    @property
    def model(self):
        return self._model
    
    @property
    def lr(self):
        return self._lr
    
    @lr.setter
    def lr(self, new_lr):
        self._lr = new_lr
        self.set_optimizer()
        
    @model.setter
    def model(self, new_model):
        self._model = new_model
        self.set_optimizer()
    
    def __call__(self, *args):
        return self.model(*args)
        
    def set_optimizer(self):
        self.optimizer = self.optimizer_fn(self.model.parameters(), lr=self.lr)
        
    
    def test_network(self, test_epochs = 50, target_loss = 0.5):
        """
        Test the model's capacity to fit the first training batch. 
        """
        for batch_to_overfit in self.data_loaders["train"]:
            break
            
        losses = train_n_epochs(model = self.model,
                       loss_fn = self.loss_fn,
                       optimizer=self.optimizer,
                       epochs=1,
                       loader=[batch_to_overfit]*test_epochs,
                        device = self.device)
        
        assert self.eval_loss([batch_to_overfit]) < target_loss, f"The model was not able to overfit on a single batch, something is wrong."
        clear_output()
        print("Your model is running correctly. Rejoice ! :-)")
        
    @torch.no_grad()
    def eval_loss(self, loader = None):
        """
        Compute the loss on a subset, default "val".
        """
        loader = self.data_loaders["val"] if loader is None else loader
        epoch_loss = 0
        for data in loader:
            inputs = data["image"].to(device)
            targets = data["label"].to(device)
            prediction = self.model(inputs)
            loss = self.loss_fn(prediction, targets)
            epoch_loss += loss.item()
        return epoch_loss
    
    def train_eval(self, epochs_per_lr, lrs):
        for lr in lrs:
            for epoch in range(epochs_per_lr):
                self.lr = lr
                previous_state = copy.copy(self.model.state_dict())
                self.train_n_epochs(1)
                val_loss = self.eval_loss()
                print(f"val loss is {val_loss}")
                if val_loss > self.validation_losses[-1] :
                    self.model.load_state_dict(previous_state)
                    print(f"No validation improvement : Skipping  learning rate {lr}")
                    break     
                self.validation_losses.append(val_loss)
        return
    
    def train_n_epochs(self, n, device = device):
        self.epoch_losses += train_n_epochs(self.model,
                       self.loss_fn,
                       self.optimizer,
                       n,
                       self.data_loaders["train"],
                       device)
  


    
    def create_report(self, confusion_labels, normalize_confusion = False, show_fig = True, agreement_threshold = 0.5):
        """
        Set the parameters for the metrics.
        """
        self.class_labels = confusion_labels
        self.threshold = agreement_threshold
        
        if (self.results is None) or (self.labels is None): 
            self.results, self.labels = results_labels_for_subset(self.model,
                                                             self.data_loaders,
                                                             "test",
                                                             device)
            self.results = self.results.numpy()
            self.labels = self.labels.numpy()
        
        self.threshold = agreement_threshold
        if show_fig :
            self.show_results()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
       
                


#Transformations for the pickle data_loader
def trim(trim_size):
    def batch_trimmer(batch_to_trim):
        batch_to_trim["image"] = batch_to_trim["image"][:,:-trim_size]
        return batch_to_trim
    return batch_trimmer

def max_pool(n_classes=9):
    def batch_max_pooler(batch_to_maxpool):
        batch_to_maxpool["label"] = nn.functional.one_hot(batch_to_maxpool["label"].argmax(1), n_classes)
        return batch_to_maxpool
    return batch_max_pooler

@torch.no_grad()
def results_labels_for_subset(model, data_loaders, subset, device):
    model.eval()
    results = torch.tensor([])
    all_labels = torch.tensor([])
    for batch in tqdm_notebook(data_loaders[subset]):
        img = batch['image']
        labels = batch['label']
        logits = model(img).cpu()
        results = torch.cat([results, logits])
        all_labels = torch.cat([all_labels, labels])
    return results, all_labels


def train_n_epochs(model, loss_fn, optimizer, epochs, loader, device):
    model = model.train().to(device)
    epoch_loss = 0
    result_tracker = []
    for epoch in range(epochs):
        for data in tqdm_notebook(loader):
            inputs = data["image"].to(device)
            targets = data["label"].to(device)
            optimizer.zero_grad()
            prediction = model(inputs)
            loss = loss_fn(prediction, targets)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result_tracker.append(epoch_loss)
        epoch_loss = 0
    return result_tracker


