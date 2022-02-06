import os
import os.path as osp
import argparse
import pickle
from ast import literal_eval
import ast
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch

global device 
device = "cuda" if torch.cuda.is_available() else "cpu"

def create_dataset(model, data_loader, save_path, recreate=False):
    """
    Save the feature vectors optained by passing them in the model at the save_path location.
    If recreate is set to True the existing folder is overwritten.
    """
    model.eval()
    if not os.path.exists(save_path) :
        if recreate : os.mkdir(save_path)
        else :
            print(f"dataset already exists at path {save_path}")
            return

    for i, batch in tqdm(enumerate(data_loader)):
        img = batch['image'].to(device)
        image_features = model(img)
        batch["image"] = image_features
        with open(f"{save_path}/batch{i}.bin","wb") as f:
            pickle.dump(batch, f)
        
        #Manage gpu memory
        del(img, batch, image_features)
        torch.cuda.empty_cache()
        
        
class Pickle_data_loader :
    
    def __init__(self, path, transformations=None):
        self.path = path
        self.transformations = transformations if transformations is not None else []
        
    def load_label(self, label, batch_size):
        batch_id, label = divmod(label, batch_size)
        batch = self._open_pickle(batch_id)
        return {key : values[label] for key, values in batch.items()}
        
    def __iter__(self):
        for i in range(len(os.listdir(self.path))):
            yield self._open_pickle(i)
    
    def _open_pickle(self, i):
        with open(osp.join(self.path, f"batch{i}.bin"),"rb") as f:
            batch = pickle.load(f)
            for transformation in self.transformations:
                batch = transformation(batch)
            return batch
        
    def load_batch(self, batch_number):
        return self._open_pickle(batch_number)
        
            
    def add_transformation(self, new_transformation):
        self.transformations.append(new_transformation)
        


def get_loaders(path, transformations=None):
    return dict([[subset, Pickle_data_loader(osp.join(path, subset), transformations)] for subset in os.listdir(path)])