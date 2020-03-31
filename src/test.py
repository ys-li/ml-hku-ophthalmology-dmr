if __name__ != '__main__':
    exit()

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
        
# ML libraries required
from fastai import *
from fastai.vision import *
from fastai.metrics import KappaScore # solution evaluated with qudratic kappa
from fastai.tabular import * # for ensemble model training
import torch
# efficientnet is not integrated into fastai yet


# Other libraries required
import matplotlib.pyplot as plt
from models.efficientnet_pytorch import EfficientNet

# garbage collector
import gc

import random
from datetime import datetime
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(42)

# %% [markdown]
# ### Define model details

# %%
models_config = {
    "effnet-b3": {
        "pretrained_name": "efficientnet-b3",
        "advprop": False,  # adversarial augmentation 
        "pretrained_path": None,
        "epoch_n": 12,
        "batch_size": 16,
        "image_dim": 384,
        "lr": 1e-3,
    }, 
    "effnet-b5": {
        "pretrained_name": "efficientnet-b5",
        "advprop": False,
        "pretrained_path": None,
        "epoch_n": 20,
        "batch_size": 16,
        "image_dim": 512,
        "lr": 1e-3,
    }, 
    "effnet-b7": {
        "pretrained_name": "efficientnet-b7",
        "advprop": False,
        "pretrained_path": None,
        "epoch_n": 30,
        "batch_size": 16,
        "image_dim": 512,
        "lr": 1e-3,
    }, 
    "effnet-b8": {
        "pretrained_name": "efficientnet-b8",
        "advprop": True,  # adversarial augmentation 
        "pretrained_path": None,
        "epoch_n": 30,
        "batch_size": 16,
        "image_dim": 512,
        "lr": 1e-3,
    }, 
}

# %% [markdown]
# ### Import preprocessing modules

# %%
import cv2
import numpy as np
from fastai.vision import pil2tensor, Image

def info_image(im):
    # Compute the center (cx, cy) and radius of the eye
    cy = im.shape[0]//2
    midline = im[cy,:]
    midline = np.where(midline>midline.mean()/3)[0]
    if len(midline)>im.shape[1]//2:
        x_start, x_end = np.min(midline), np.max(midline)
    else: # This actually rarely happens p~1/10000
        x_start, x_end = im.shape[1]//10, 9*im.shape[1]//10
    cx = (x_start + x_end)/2
    r = (x_end - x_start)/2
    return cx, cy, r

def resize_image(im, image_dim=None, augmentation=False):
    # Crops, resizes and potentially augments the image to image_dim
    cx, cy, r = info_image(im)
    scaling = image_dim/(2*r)
    rotation = 0
    if augmentation:
        scaling *= 1 + 0.3 * (np.random.rand()-0.5)
        rotation = 360 * np.random.rand()
    M = cv2.getRotationMatrix2D((cx,cy), rotation, scaling)
    M[0,2] -= cx - image_dim/2
    M[1,2] -= cy - image_dim/2
    return im # disbale resizing
    # return cv2.warpAffine(im,M,(image_dim,image_dim)) # This is the most important line

def crop_image_from_gray(img,tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """  
    if img.ndim == 2: # gray scale
        mask = img>tol  # create an array (mask) of pixels higher than tolerance
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim == 3: # color image
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert to gray
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else: 
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

PARAM = 96
def Radius_Reduction(img,PARAM):
    h,w,c=img.shape
    Frame=np.zeros((h,w,c),dtype=np.uint8)
    cv2.circle(Frame,(int(math.floor(w/2)),int(math.floor(h/2))),int(math.floor((h*PARAM)/float(2*100))), (255,255,255), -1)
    Frame1=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    img1 =cv2.bitwise_and(img,img,mask=Frame1)
    return img
    
def contrast_and_crop(torch_tensor, image_dim=None, path=None, sigmaX=10, median_blur=True):
    if path is None:
        np_image = image2np(torch_tensor) * 255 # convert tensor image to numpy array
        np_image = np_image.astype(np.uint8)
    else:
        np_image = cv2.imread(path)  # may return a string, in which case treat it as a path
    
    image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (image_dim, image_dim))
    image = resize_image(image, image_dim)
    
    if median_blur:
        k = np.max(image.shape)//20*2+1
        blur = cv2.medianBlur(image, k)
    else:
        blur = cv2.GaussianBlur( image , (0,0) , sigmaX) 
        
    image = cv2.addWeighted ( image,4, blur,-4 ,128)

    image = Radius_Reduction(image, PARAM)

    return pil2tensor(image, np.float32).div_(255) # return tensor

def advprop_normalise(torch_tensor):
    np_image = image2np(torch_tensor) * 2.0 - 1.0 
    return pil2tensor(np_image, np.float32) # return tensor

# we later override the load image function of fast.ai to make sure all images
# including train and test sets, are properly processed
# define function    

# def _load_format(path, convert_mode, after_open, image_dim)->Image:
#     image = contrast_and_crop(None, image_dim, path, median_blur=False) 
# 
#     return Image(image) # return fastai Image format

# %% [markdown]
# ### Load data

# %%
class PreProcessCommonWrapper(object):
    def __init__(self, image_dim):
        self.image_dim = image_dim
        self.__name__ = "PreProcessCommonWrapper"
        self.__annotations__ = {}
    def __call__(self, t): # the function formerly known as "bar"
        return contrast_and_crop(t, self.image_dim)

def load_data(model_config):
    current_model_config = model_config
    base19_dir = os.path.join('../', 'input/aptos_image/')
    train19_dir = os.path.join('../', 'input/aptos_image/train_19/')
    base15_dir = os.path.join('../', 'input/aptos_image/')
    train15_dir = os.path.join('../', 'input/aptos_image/train_15/')
    
    df = pd.read_csv(os.path.join(base19_dir, 'labels/trainLabels19.csv'))
    
    df15 = pd.read_csv(os.path.join(base15_dir, 'labels/trainLabels15.csv'))
    
    # change id_code to accessible path and drops the id_code col
    df['path'] = df['file_name'].map(lambda x: os.path.join(train19_dir,f'{x}.jpg'))
    df = df.drop(columns=['file_name'])
    df15['path'] = df15['file_name'].map(lambda x: os.path.join(train15_dir,f'{x}.jpg'))
    df15 = df15.drop(columns=['file_name'])
    
    # add extras to training set
    df = pd.concat([df,df15], ignore_index=True)
    
    src = ImageList.from_df(df=df, path = './', cols='path')                    .split_by_rand_pct()                    .label_from_df(cols='diagnosis', label_cls=FloatList)  # although labels are in integer form, they are intepreted as Float for training purposes
            
    transformations = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.3,max_lighting=0.1,p_lighting=0.5)
    
    # custom pre-processing (contrast and crop)
    pre_process_common_wrapper = PreProcessCommonWrapper(model_config["image_dim"])
    pre_process_ccs = [TfmPixel(pre_process_common_wrapper)()]
    advprop = model_config["advprop"]
    if advprop:
        pre_process_ccs.append(TfmPixel(advprop_normalise)())
    # apply transformations to training set, but apply the pre_process to train and valid set
    tfms = [transformations[0] + pre_process_ccs, transformations[1] + pre_process_ccs]
    
    # transform data sets
    data = src.transform(tfms, size=model_config["image_dim"], resize_method=ResizeMethod.CROP,padding_mode='zeros',)               .databunch(bs=model_config["batch_size"], num_workers=4)               .normalize(imagenet_stats if not advprop else None) # default normalise with imagenet stats, prebuilt into fast.ai library    
    
    print("loaded data")
    return (df, data)


# %%
# lets visualise what we have got
# df, data = load_data(models_config["effnet-b3"])
# data.show_batch(rows=3, figsize=(10,10))

# %% [markdown]
# ### Helper functions in loading models

# %%
def getModel(model_name, data, model_dir=None, advprop=False, **kwargs):
    from os.path import abspath
    if model_dir is not None:
        model_dir = abspath(model_dir)
    model = EfficientNet.from_pretrained(model_name, advprop=advprop)
    model._fc = nn.Linear(model._fc.in_features,data.c) # .c returns number of output cells, ._fc returns the module
    return model

def get_learner(model_name, data, model_dir=None):
    return Learner(data, getModel(model_name, data, model_dir=model_dir), metrics = [quadratic_kappa])            .mixup()            .to_fp16() 

# quadratic kappa score
from sklearn.metrics import cohen_kappa_score
def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')


# %%
def main():
    for config_name in models_config:
        print(f"---- TRAINING STARTING FOR {config_name} ---")
        config = models_config[config_name]
        df, data = load_data(config)
        learner = get_learner(config["pretrained_name"], data, model_dir=config["pretrained_path"])
        lr = config["lr"]
        learner.lr_find()
    # learner.recorder.plot()   
    # learner.fit_one_cycle(1, lr) # config['epoch_n']
    # learner.save(f'{config_name}_{int(datetime.now().timestamp())}')

if __name__ == '__main__':
    main()


# %%


