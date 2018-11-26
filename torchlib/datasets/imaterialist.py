
import os
import numpy as np

from pytvision.datasets import utility 
from pytvision.datasets.imageutl import imageProvide
from pytvision.transforms.aumentation import ObjectImageTransform

import torch
from torchvision import datasets


class IMaterialistDatset(object):
    
    def __init__(self, pathname, subset, ext, transform=None ): 
            
        data = datasets.ImageFolder( os.path.join( pathname, subset ) )
        
        self.data = data
        self.labels = np.array([ data.imgs[i][1] for i in range(len(data)) ])        
        self.classes = data.classes 
        self.class_to_idx = data.class_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
           


class IMaterialistImageDataset(object):
    
    def __init__(self, pathname, ext, num_channels=1, transform=None ):       
        self.data = imageProvide(pathname, ext )        
        self.transform = transform   
        self.num_channels = num_channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):   
        image = self.data[idx]        
        Id = idx
        
        image = np.array(image) 
        image = utility.to_channels(image, self.num_channels)      
    
        obj = ObjectImageTransform( image )
        if self.transform: 
            obj = self.transform(obj)        

        Id = torch.from_numpy( np.array([Id])).float()
        return Id, obj.to_value() 
    
    def getId(self, i ):
        return self.data.getid(i)



    
class IMaterialist(imageProvide):
    '''
    Management dataset <images, labes>
    '''
    def __init__(self, path, subset = 'train' ):
        super(IMaterialist, self).__init__( path, 'jpg', subset )

        self.labels_orgs = np.array([ int(y.split('.')[0].split('_')[-1] ) for y in self.data  ]) 
        self.classes_orgs  = np.unique( self.labels_orgs )

        #retificate class
        classes = np.arange( len(self.labels_orgs) )
        labels  = np.zeros_like( self.labels_orgs )
        for i,c in enumerate(self.classes_orgs):
            indexs = np.where(self.labels_orgs==c)[0]
            labels[indexs] = i     
           
        self.labels = labels
        self.classes = classes
    
    def __getitem__(self, i):
        image = self.getimage(i)
        label = self.labels[i] 
        return image, label

