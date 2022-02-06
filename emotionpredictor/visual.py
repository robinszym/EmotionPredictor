import os
import pickle

import seaborn as sns
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import clip

from tqdm import tqdm


ARTEMIS_EMOTIONS = ['amusement',
 'awe',
 'contentment',
 'excitement',
 'anger',
 'disgust',
 'fear',
 'sadness',
 'something else']


def emotion_colors(palette = sns.palettes.color_palette("colorblind")):
    return palette[:3]+[palette[-1]]+[palette[3]] + palette[5:10]
    
def standardized_fig():
    plt.figure(figsize = (16,9))
    plt.grid(axis = "y", ls = ":", alpha = 0.8)
    
def show_torch_image(torch_image, axis = None):
    torch_image = torch_image.detach().cpu()
    c=torch_image-torch_image.min()
    c/= c.max()
    if axis is None :
        plt.imshow(c.permute(1, 2, 0))
        plt.axis("off")
    else :
        axis.imshow(c.permute(1, 2, 0))
        axis.axis("off")
    return
        
def tensor2img(tensor):
    tensor = tensor.detach().cpu()
    c = tensor-tensor.min()
    c /= c.max()
    return c.permute(1,2,0)

def save_fig(path):
    a = plt.gcf()
    a.savefig(path, bbox_inches = "tight")
    return

