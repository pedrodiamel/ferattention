
import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as TF
from torchvision import transforms, utils

from pytvision.transforms.aumentation import  ObjectImageMetadataTransform
from pytvision.transforms import transforms as mtrans

from torchlib.transforms import functional as F
from torchlib.datasets.synthetic_fer  import SyntheticFaceDataset
from torchlib.datasets.factory  import FactoryDataset
from torchlib.datasets.datasets import Dataset
from torchlib.attentionnet import AttentionNeuralNet

import skimage.morphology as morph
import scipy.ndimage as ndi


project          = 'out'
name             = 'attention_atentionresnet34_mcedice_adam_coco_003'
pathnamedataset  = '~/.datasets'
pathnewdataset   = 'netruns/newdataset'
pathmodel        = os.path.join( project, name, 'models/model_best.pth.tar' )
pathproject      = os.path.join( project, name )
batch_size       = 2
workers          = 1
cuda             = False
parallel         = False
gpu              = 1
seed             = 1
imsize           = 128

namedataset = FactoryDataset.ck
subset = FactoryDataset.training


def norm(x):
    x = x-x.min()
    x = x / x.max()
    return x

def create_dataset():
    
    # Load dataset
    print('>> Load dataset ...')
    dataset = Dataset(    
        data=FactoryDataset.factory(
            pathname=pathnamedataset, 
            name=namedataset, 
            subset=subset, 
            #transform=train_transform, 
            download=True 
        ),
        num_channels=3,
        transform=transforms.Compose([
                mtrans.ToResize( (imsize,imsize), resize_mode='square' ),
                #mtrans.RandomCrop( (255,255), limit=50, padding_mode=cv2.BORDER_CONSTANT  ),
                #mtrans.ToResizeUNetFoV(imsize, cv2.BORDER_REFLECT_101),
                mtrans.ToTensor(),
                mtrans.ToMeanNormalization( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
                #mtrans.ToNormalization(),
                ])
        )

    print(len(dataset))
    print( dataset.classes )
    print( dataset.data.class_to_idx )
    
    # load model
    print('>> Load model ...')

    net = AttentionNeuralNet( 
        patchproject=project, 
        nameproject=name, 
        no_cuda=cuda, 
        parallel=parallel, 
        seed=seed, 
        gpu=gpu 
        )

    if net.load( pathmodel ) is not True:
        assert(False)

    # create data
    print('>> Create dataset ...')
    for idx in range(  len( tqdm(dataset) ) ):
        
        sample = dataset[ idx ]    
        image = sample['image'].unsqueeze(0)
        label = sample['label'].argmax()
        #image = F.fliplr( image )
        
        z, y_lab_hat, y_mask_hat, att = net( image )
        score = TF.softmax( y_mask_hat, dim=1 ).float()
        score = score.data.cpu().numpy().transpose(2,3,1,0)[...,0]
        pred  = np.argmax( score, axis=2 )

        #att = att.data.cpu().numpy().transpose(2,3,1,0)[...,0]
        #att = norm(att)

        image = image.data.cpu().numpy().transpose(2,3,1,0)[...,0]
        y_lab_hat = y_lab_hat.argmax()

        image = norm(image)
        mask = pred 
        
        # morph 
        struct_el = morph.disk(11)
        mask = ndi.morphology.binary_fill_holes( mask )
        # mask = morph.binary_closing(mask, selem=struct_el)
        mask = morph.binary_opening(mask, selem=struct_el)

        mask_ex = np.stack( (mask, mask, mask),axis=2 )
        im_mask = (mask_ex)*image + (1-mask_ex)*np.zeros_like(image)

        # save
        cv2.imwrite('{}/{:06d}_{}.png'.format(pathnewdataset,idx,label), (im_mask*255.0).astype(np.uint8) )
                



if __name__ == '__main__':
    create_dataset()