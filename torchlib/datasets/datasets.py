
import os
import numpy as np
import random
from collections import namedtuple

import torch
from pytvision.datasets import utility 
from pytvision.datasets.imageutl import imageProvide
from pytvision.transforms.aumentation import ObjectImageAndLabelTransform, ObjectImageTransform

import warnings
warnings.filterwarnings("ignore")




class Dataset( object ):
    """
    Generic dataset
    """

    def __init__(self, 
        data,
        num_channels=1,
        count=None,
        transform=None 
        ):
        """
        Initialization 
        Args:
            @data: dataprovide class
            @num_channels: 
            @tranform: tranform           
        """             
        
        if count is None: count = len(data)
        self.count = count
        self.data = data
        self.num_channels = num_channels        
        self.transform = transform   
        self.labels = data.labels
        self.classes = np.unique(self.labels) 
        self.numclass = len(self.classes)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        image, label = self.data[idx]
        image = np.array(image) 
        image = utility.to_channels(image, self.num_channels)        
        label = utility.to_one_hot(label, self.numclass)

        obj = ObjectImageAndLabelTransform( image, label )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()
    

class ResampleDataset( object ):
    """
    Resample data for generic dataset
    """

    def __init__(self, 
        data,
        num_channels=1,
        count=200,
        transform=None  
        ):
        """
        Initialization   
        data: dataloader class
        tranform: tranform           
        """             
        
        self.num_channels=num_channels
        self.data = data        
        self.transform = transform   
        self.labels = data.labels 
        self.count=count
        
        #self.classes = np.unique(self.labels)
        self.classes, self.frecs = np.unique(self.labels, return_counts=True)
        self.numclass = len(self.classes)
        
        #self.weights = 1-(self.frecs/np.sum(self.frecs))
        self.weights = np.ones( (self.numclass,1) )        
        self.reset(self.weights)
        
        self.labels_index = list()
        for cl in range( self.numclass ):             
            indx = np.where(self.labels==cl)[0]
            self.labels_index.append(indx)            

    
    def reset(self, weights):        
        self.dist_of_classes = np.array(random.choices(self.classes, weights=weights, k=self.count ))

    def __len__(self):
        return self.count

    def __getitem__(self, idx):   
                
        idx = self.dist_of_classes[idx]
        class_index = self.labels_index[idx]
        n =  len(class_index)
        idx = class_index[ random.randint(0,n-1) ]

        image, label = self.data[idx]

        image = np.array(image) 
        image = utility.to_channels(image, self.num_channels)            
        label = utility.to_one_hot(label, self.numclass)

        obj = ObjectImageAndLabelTransform( image, label )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()

class SecuencialSamplesDataset( object ):
    """
    Generic dataset for extratificate secuencial samples
    """

    def __init__(self, 
        data,
        count=None,
        num_channels=1,
        transform=None        
        ):
        """ 
        Initialization            
        """            

        if count is None: count = len(data)
        self.num_channels=num_channels
        self.data = data
        self.num = count

        # make index
        self.labels = data.labels 
        self.classes = np.unique(self.labels)
        self.numclass = len(self.classes)

        self.labels_index = list()
        for cl in range( self.numclass ):             
            indx = np.where(self.labels==cl)[0]
            self.labels_index.append(indx)            
        self.transform = transform     
        

    def __len__(self):
        return self.num

    def __getitem__(self, idx):   

        idx = idx % self.numclass
        class_index = self.labels_index[idx]
        n =  len(class_index)
        idx = class_index[ random.randint(0,n-1) ]  
        image, label = self.data[idx]

        image = np.array(image) 
        image = utility.to_channels(image, self.num_channels)            
        label = utility.to_one_hot(label, self.numclass)

        obj = ObjectImageAndLabelTransform( image, label )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()

    
    

class SecuencialExSamplesDataset( object ):
    """
    Generic dataset for extratificate secuencial ext samples
    """

    def __init__(self, 
        data,
        count=200,
        n_set=3,
        batch_size=10,
        num_channels=1,
        transform=None        
        ):
        """ 
        Initialization            
        """            

        self.num_channels=num_channels
        self.data = data
        self.num = count
        self.n_set = n_set
        self.batch_size = batch_size

        # make index
        self.labels = data.labels 
        self.classes = np.unique(self.labels)
        self.numclass = len(self.classes)
        self.transform = transform     

        self.labels_index = list()
        for cl in range( self.numclass ):             
            indx = np.where(self.labels==cl)[0]
            self.labels_index.append(indx) 

        self.make_manifold_list()


    def __len__(self):
        return int( self.num*self.n_set*self.batch_size )

    def __getitem__(self, idx):   

        idx = self.index[ idx ]
        image, label = self.data[idx]

        image = np.array(image) 
        image = utility.to_channels(image, self.num_channels)            
        label = utility.to_one_hot(label, self.numclass)

        obj = ObjectImageAndLabelTransform( image, label )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()


    def _reset_classes(self):
        self.class_select = np.random.choice( self.classes, self.n_set, replace=False )   

    def make_manifold_list(self):        

        self._reset_classes()
        n = self.num * self.n_set * self.batch_size# iterations 

        index = np.zeros((n), dtype=int)
        for i in range( n ):
            
            if i % self.batch_size * self.n_set == 0:
                self._reset_classes()
            
            k = i % self.n_set
            class_index = self.labels_index[ self.class_select[ k ] ]
            idx = class_index[ random.randint(0, len(class_index)-1) ]
            index[i] = idx

        self.index = index
        


class TripletsDataset( object ):
    """
    TripletsDataset
    """

    def __init__(self, 
        data,
        n_triplets=100,
        num_channels=1,
        transform=None):
        """              
        """  

        self.data = data
        self.num_channels=num_channels
        
        # make triplets
        self.labels = data.labels
        self.classes, self.frecs = np.unique(self.labels, return_counts=True)
        #self.weights = np.array([ 1-(self.frecs[i]/np.sum(self.frecs)) for i in self.labels ])
        
        self.numclass = len(  self.classes )
        self.num_triplets = n_triplets
        self.make_triplet_list(n_triplets)

        self.transform = transform 

    def reset(self):
        print('Reset dataloader ...')
        self.make_triplet_list(self.num_triplets)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):   
    
        idx1, idx2, idx3 = self.triplets[idx]
        img1, lab1 = self.data[idx1]; 
        img2, lab2 = self.data[idx2]; 
        img3, lab3 = self.data[idx3]; 

        img1 = np.array(img1)
        img2 = np.array(img2)
        img3 = np.array(img3)

        img1 = utility.to_channels(img1, self.num_channels)
        img2 = utility.to_channels(img2, self.num_channels)
        img3 = utility.to_channels(img3, self.num_channels)

        lab1 = utility.to_one_hot(lab1, self.numclass)
        lab2 = utility.to_one_hot(lab2, self.numclass)
        lab3 = utility.to_one_hot(lab3, self.numclass)      

        a = ObjectImageAndLabelTransform( img1, lab1 )
        b = ObjectImageAndLabelTransform( img2, lab2 )
        c = ObjectImageAndLabelTransform( img3, lab3 )          
        if self.transform is not None:
            a = self.transform( a )
            b = self.transform( b )
            c = self.transform( c )
        
        return {'a':a.to_dict(), 'b':b.to_dict(), 'c':c.to_dict()}



    def make_triplet_list(self, ntriplets):
                       
        self.triplets = []         
        nc = self.numclass
        #choice = lambda seq: np.array([ random.choice(seq) for _ in range( int(ntriplets/nc) ) ]) 

        for cx in range(nc):          

            class_idx = cx
            # a, b, c are index of labels where it's equal to class_idx        
            a = np.array(random.choices( np.where(self.labels==class_idx)[0], k=int(ntriplets/nc) ))
            b = np.array(random.choices( np.where(self.labels==class_idx)[0], k=int(ntriplets/nc) ))

            #while np.any((a-b)==0): #aligning check
            while np.sum((a-b) == 0 )/b.shape[0] > 0.1: #aligning check
                random.shuffle(b)                                
            
            #index = np.where(self.labels!=class_idx)[0]         
            #w = self.weights[index]
            #c = np.array(random.choices(index, weights=w, k=int(ntriplets/nc) ))
            
            c = np.array(random.choices(np.where(self.labels!=class_idx)[0], k=int(ntriplets/nc) ))            
            self.triplets += zip(a,b,c)
    
        random.shuffle(self.triplets)
        self.num_triplets = (ntriplets/nc)*nc

        

    def regenerate_triplet_list(self, sampler, frac_hard):
                
        # negatives is a tuple of anchors and negative examples
        num_random_triplets = self.num_triplets*(1.0-frac_hard)
        # adjust number of random triplets so that it is a multiple of num_classes
        num_random_triplets = int(math.ceil(num_random_triplets)/self.num_classes)*self.num_classes
        num_hard = self.num_triplets - num_random_triplets
        
        print("Number of hard triplets %d ..." % num_hard)

        self.make_triplet_list(num_random_triplets)
        neg_hard_examples = sampler.ChooseNegatives(num_hard)
        # choose random positives (for now atleast) for hard negatives
        for pair in neg_hard_examples:
            a, c = pair
            anchor_class = self.labels[a]
            b = np.random.choice(np.where(self.labels == anchor_class)[0])
            self.triplets.append((a, b, c))
        np.random.shuffle(self.triplets)

