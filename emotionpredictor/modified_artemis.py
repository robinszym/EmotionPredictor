import argparse
import ast
import os.path as osp

import torch
import pandas as pd
import numpy as np

from artemis.in_out.datasets import ImageClassificationDataset


def max_io_workers():
    """return all/max possible available cpus of machine."""
    return max(mp.cpu_count() - 1, 1)

def get_original_wikiart_dataloaders(path_emotion_histogram_csv, artemis_preprocessed_dir, wikiart_path, preprocess):
    """
    Code extracted from the notebook found in the Artemis repo :
        https://github.com/optas/artemis/blob/master/artemis/notebooks/deep_nets/emotions/image_to_emotion_classifier.ipynb
    """
    print("debug")
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
