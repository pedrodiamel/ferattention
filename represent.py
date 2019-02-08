
import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt 
from tqdm import tqdm

import torch
import torch.nn.functional as TF
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from pytvision.transforms.aumentation import  ObjectImageMetadataTransform
from pytvision.transforms import transforms as mtrans

sys.path.append('../')
from torchlib.transforms import functional as F
from torchlib.datasets.fersynthetic  import SyntheticFaceDataset
from torchlib.datasets.factory  import FactoryDataset
from torchlib.datasets.datasets import Dataset
from torchlib.datasets.fersynthetic  import SyntheticFaceDataset

from torchlib.attentionnet import AttentionNeuralNet
from aug import get_transforms_aug, get_transforms_det


# METRICS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics


from argparse import ArgumentParser


def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('--project',     metavar='DIR',  help='path to projects')
    parser.add_argument('--projectname', metavar='DIR',  help='name projects')
    parser.add_argument('--pathdataset', metavar='DIR',  help='path to dataset')
    parser.add_argument('--namedataset', metavar='S',    help='name to dataset')
    parser.add_argument('--pathnameout', metavar='DIR',  help='path to out dataset')
    parser.add_argument('--filename',    metavar='S',    help='name of the file output')
    parser.add_argument('--model',       metavar='S',    help='filename model')  
    return parser


def main():
    
    parser = arg_parser();
    args = parser.parse_args();

    # Configuration
    project         = args.project
    projectname     = args.projectname
    pathnamedataset = args.pathdataset  
    pathnamemodel   = args.model
    pathproject     = os.path.join( project, projectname )
    pathnameout     = args.pathnameout
    filename        = args.filename
    namedataset     = args.namedataset 
    
    no_cuda=False
    parallel=False
    gpu=0
    seed=1  
    imagesize=128
    batch_size=100
    idenselect=[]
        
    # experiments
    experiments = [ 
        { 'name': namedataset, 'subset': FactoryDataset.training,   'real': True },
        ]
        
    # Load models
    print('>> Load model ...')
    network = AttentionNeuralNet(
        patchproject=project,
        nameproject=projectname,
        no_cuda=no_cuda,
        parallel=parallel,
        seed=seed,
        gpu=gpu,
        )

    cudnn.benchmark = True

    # load model
    if network.load( pathnamemodel ) is not True:
        print('>>Error!!! load model')
        assert(False)
        
    size_input = network.size_input
    for  i, experiment in enumerate(experiments):
            
        name_dataset = experiment['name']
        subset = experiment['subset']
        breal = experiment['real']
        dataset = []
                
        # real dataset 
        dataset = Dataset(    
            data=FactoryDataset.factory(
                pathname=pathnamedataset, 
                name=namedataset, 
                subset=subset, 
                idenselect=idenselect,
                download=True 
            ),
            num_channels=3,
            transform=get_transforms_det( imagesize ),
            )
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10 )
            
        print(breal)
        print(subset)
        #print(dataloader.dataset.data.classes)
        print(len(dataset))
        print(len(dataloader))
            
        # representation 
        Y_labs, Y_lab_hats, Zs = network.representation( dataloader, breal )            
        print(Y_lab_hats.shape, Zs.shape, Y_labs.shape)
            
        reppathname = os.path.join( pathproject, 'rep_{}_{}_{}_{}.pth'.format(projectname, namedataset, subset, 'real' if breal else 'no_real' ) )
        torch.save( { 'Yh':Y_lab_hats, 'Z':Zs, 'Y':Y_labs }, reppathname )
        print( 'save representation ...' )
        
    
    print('DONE!!!')
        

if __name__ == '__main__':
    main()