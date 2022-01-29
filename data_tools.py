import torch
import os
import pandas as pd
import os.path as osp
from artemis.in_out.datasets import ImageClassificationDataset
import argparse
import pickle
from ast import literal_eval
import ast
from tqdm import tqdm
import numpy as np

global device 
device = "cuda" if torch.cuda.is_available() else "cpu"

def create_dataset(model, data_loader, save_path, recreate = False):
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
    
    def __init__(self, path, transformations = None ):
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
        

def max_io_workers():
    """return all/max possible available cpus of machine."""
    return max(mp.cpu_count() - 1, 1)

def get_original_wikiart_dataloaders(path_emotion_histogram_csv, artemis_preprocessed_dir, wikiart_path, preprocess):
    """
    Code extracted from the notebook found in the Artemis repo :
        https://github.com/optas/artemis/blob/master/artemis/notebooks/deep_nets/emotions/image_to_emotion_classifier.ipynb
    """
    image_hists = pd.read_csv(path_emotion_histogram_csv)
    image_hists.emotion_histogram = image_hists.emotion_histogram.map(ast.literal_eval)
    image_hists.emotion_histogram = image_hists.emotion_histogram.apply(lambda x: (np.array(x) / float(sum(x))).astype('float32'))
    GPU_ID = 0 
    artemis_data = pd.read_csv(osp.join(artemis_preprocessed_dir, 'artemis_preprocessed.csv'))
    artemis_data = artemis_data.drop_duplicates(subset=['art_style', 'painting'])
    artemis_data.reset_index(inplace=True, drop=True)

    # keep only relevant info + merge
    artemis_data = artemis_data[['art_style', 'painting', 'split']] 
    artemis_data = artemis_data.merge(image_hists)
    artemis_data = artemis_data.rename(columns={'emotion_histogram': 'emotion_distribution'})
    n_emotions = len(image_hists.emotion_histogram[0])
    assert all(image_hists.emotion_histogram.apply(len) == n_emotions)
    parser = argparse.ArgumentParser() # use for convenience instead of say a dictionary
    args = parser.parse_args([])

    # deep-net data-handling params. note if you want to reuse this net with neural-speaker 
    # it makes sense to keep some of the (image-oriented) parameters the same accross the nets.
    args.lanczos = True
    args.img_dim = 256
    args.num_workers = 8
    args.batch_size = 20
    args.gpu_id = 0

    args.img_dir = wikiart_path
    
    dataloaders, _ = modified_image_emotion_distribution_df_to_pytorch_dataset(df = artemis_data,args = args, preprocess=preprocess)
    return dataloaders

# Reimplementation of the image_emotion_distribution_df_to_pytorch_dataset to accept a preprocess argument on
# the image transform. 
def modified_image_emotion_distribution_df_to_pytorch_dataset(df, args, preprocess, drop_thres=None):
    """ Convert the pandas dataframe that carries information about images and emotion (distributions) to a
    dataset that is amenable to deep-learning (e.g., for an image2emotion classifier).
    :param df:
    :param args:
    :param drop_thres: (optional, float) if provided each distribution of the training will only consist of examples
        for which the maximizing emotion aggregates more than this (drop_thres) mass.
    :return: pytorch dataloaders & datasets
    """
    dataloaders = dict()
    datasets = dict()
    #only this line is changed from the original implementation
    img_transforms = preprocess

    if args.num_workers == -1:
        n_workers = max_io_workers()
    else:
        n_workers = args.num_workers

    for split, g in df.groupby('split'):
        g.reset_index(inplace=True, drop=True)

        if split == 'train' and drop_thres is not None:
            noise_mask = g['emotion_distribution'].apply(lambda x: max(x) > drop_thres)
            print('Keeping {} of the training data, since for the rest their emotion-maximizer is too low.'.format(noise_mask.mean()))
            g = g[noise_mask]
            g.reset_index(inplace=True, drop=True)


        img_files = g.apply(lambda x : osp.join(args.img_dir, x.art_style,  x.painting + '.jpg'), axis=1)
        img_files.name = 'image_files'

        dataset = ImageClassificationDataset(img_files, g.emotion_distribution,
                                             img_transform=img_transforms)

        datasets[split] = dataset
        b_size = args.batch_size if split=='train' else args.batch_size * 2
        dataloaders[split] = torch.utils.data.DataLoader(dataset=dataset,
                                                         batch_size=b_size,
                                                         shuffle=split=='train',
                                                         num_workers=n_workers)
    return dataloaders, datasets

def get_loaders(path, transformations):
    return dict([[subset, Pickle_data_loader(osp.join(path, subset), transformations)] for subset in os.listdir(path)])