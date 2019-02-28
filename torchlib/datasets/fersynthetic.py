


import os
import numpy as np
import cv2
import random

import torch
import torch.utils.data as data
import torch.nn.functional


from ..transforms.ferrender import Generator

from pytvision.datasets import imageutl as imutl
from pytvision.datasets import utility
from pytvision.transforms import functional as F


from pytvision.transforms.aumentation import(     
     ObjectImageMaskAndWeightTransform, 
     ObjectImageTransform, 
     ObjectImageAndLabelTransform, 
     ObjectImageAndMaskTransform, 
     ObjectRegressionTransform, 
     ObjectImageAndAnnotations,
     ObjectImageAndMaskMetadataTransform,
    )


import warnings
warnings.filterwarnings("ignore")


class SyntheticFaceDataset( data.Dataset ):
    '''
    Management for Synthetic Face dataset
    '''
    generate_image = 'image'
    generate_image_and_mask = 'image_and_mask' 


    def __init__(self, 
        data,
        pathnameback=None,
        ext='jpg',
        count=None,
        num_channels=3,
        generate='image_and_mask',
        iluminate=True, angle=45, translation=0.3, warp=0.1, factor=0.2,
        transform_image=None,
        transform_data=None,
        ):
        """Initialization           
        """            
              
        self.data = data
        self.bbackimage = pathnameback != None
        self.databack = None

        if count is None:
            count = len(data)
        
        if self.bbackimage: 
            pathnameback  = os.path.expanduser( pathnameback )
            self.databack = imutl.imageProvide( pathnameback, ext=ext ) 
        
        self.num_classes=data.numclass
        self.labels = data.labels
        self.num_channels = num_channels
        self.generate = generate
        self.ren = Generator( iluminate, angle, translation, warp, factor )        
        self.count=count
        
        self.transform_image = transform_image 
        self.transform_data = transform_data 
       
  

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        # read image 
        image, label = self.data[ (idx)%len(self.data)  ]
        #A,A_inv = F.compute_norm_mat( image.shape[1], image.shape[0] )
        #image = F.equalization(image,A,A_inv)
        image = utility.to_channels(image, self.num_channels)
        
        # read background 
        if self.bbackimage:
            idxk = random.randint(1, len(self.databack) - 1 )
            back = self.databack[ idxk  ] 
            back = F.resize_image(back, 640, 1024, resize_mode='crop', interpolate_mode=cv2.INTER_LINEAR);
            back = utility.to_channels(back, self.num_channels)
        else:
            back = np.ones( (640,1024,3), dtype=np.uint8 )*255
       
        if self.generate == 'image':
            obj = ObjectImageTransform( image  )
            
        elif self.generate == 'image_and_mask':                           
            
            image_org, image_ilu, mask, h = self.ren.generate( image, back )  
                        
            image_org = utility.to_gray( image_org.astype(np.uint8)  )
            image_org = utility.to_channels(image_org, self.num_channels)
            image_org = image_org.astype(np.uint8)
            
            image_ilu = utility.to_gray( image_ilu.astype(np.uint8)  )
            image_ilu = utility.to_channels(image_ilu, self.num_channels)
            image_ilu = image_ilu.astype(np.uint8) 
                               
            mask = mask[:,:,0]
            mask_t = np.zeros( (mask.shape[0], mask.shape[1], 2) )
            mask_t[:,:,0] = (mask == 0).astype( np.uint8 ) # 0-backgraund
            mask_t[:,:,1] = (mask == 1).astype( np.uint8 )
                        
            obj_image = ObjectImageTransform( image_org.copy()  )
            obj_data = ObjectImageAndMaskMetadataTransform( image_ilu.copy(), mask_t, np.concatenate( ( [label], h),axis=0 ) ) #np.array([label])
                        
        else: 
            assert(False)         

        if self.transform_image: 
            obj_image = self.transform_image( obj_image ) 

        if self.transform_data: 
            obj_data = self.transform_data( obj_data )
            
        x_img, y_mask, y_lab = obj_data.to_value()
        x_org = obj_image.to_value()
        
        return x_org, x_img, y_mask, y_lab


class SecuencialSyntheticFaceDataset( data.Dataset ):
    '''
    Management for Synthetic Face dataset
    '''
    generate_image = 'image'
    generate_image_and_mask = 'image_and_mask' 


    def __init__(self, 
        data,
        pathnameback=None,
        ext='jpg',
        count=None,
        num_channels=3,
        generate='image_and_mask',
        iluminate=True, angle=45, translation=0.3, warp=0.1, factor=0.2,
        transform_image=None,
        transform_data=None,
        ):
        """Initialization           
        """            
              
        self.data = data
        self.bbackimage = pathnameback != None
        self.databack = None

        if count is None:
            count = len(data)
        
        if self.bbackimage: 
            pathnameback  = os.path.expanduser( pathnameback )
            self.databack = imutl.imageProvide( pathnameback, ext=ext ) 
        
        self.num_classes=data.numclass
        self.labels = data.labels
        self.num_channels = num_channels
        self.generate = generate
        self.ren = Generator( iluminate, angle, translation, warp, factor )        
        self.count=count
        
        self.transform_image = transform_image 
        self.transform_data = transform_data 
                
        self.labels_index = list()
        for cl in range( self.num_classes ):             
            indx = np.where( self.labels==cl )[0]
            self.labels_index.append( indx )     
        
        
    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        # read image         
        idx = idx % self.num_classes        
        class_index = self.labels_index[ idx ]
        n =  len( class_index )        
        idx = class_index[ random.randint(0,n-1) ]  
        image, label = self.data[ idx ]  
        
        #image, label = self.data[ (idx)%len(self.data)  ]        
        
        #A,A_inv = F.compute_norm_mat( image.shape[1], image.shape[0] )
        #image = F.equalization(image,A,A_inv)
        image = utility.to_channels(image, self.num_channels)
        
        # read background 
        if self.bbackimage:
            idxk = random.randint(1, len(self.databack) - 1 )
            back = self.databack[ idxk  ] 
            back = F.resize_image(back, 640, 1024, resize_mode='crop', interpolate_mode=cv2.INTER_LINEAR);
            back = utility.to_channels(back, self.num_channels)
        else:
            back = np.ones( (640,1024,3), dtype=np.uint8 )*255
       
        if self.generate == 'image':
            obj = ObjectImageTransform( image  )
            
        elif self.generate == 'image_and_mask':   
                        
            
            image_org, image_ilu, mask, h = self.ren.generate( image, back )  
            
            image_org = utility.to_gray( image_org.astype(np.uint8)  )
            image_org = utility.to_channels(image_org, self.num_channels)
            image_org = image_org.astype(np.uint8)
            
            image_ilu = utility.to_gray( image_ilu.astype(np.uint8)  )
            image_ilu = utility.to_channels(image_ilu, self.num_channels)
            image_ilu = image_ilu.astype(np.uint8) 
                               
            mask = mask[:,:,0]
            mask_t = np.zeros( (mask.shape[0], mask.shape[1], 2) )
            mask_t[:,:,0] = (mask == 0).astype( np.uint8 ) # 0-backgraund
            mask_t[:,:,1] = (mask == 1).astype( np.uint8 )
                        
            obj_image = ObjectImageTransform( image_org.copy()  )
            obj_data = ObjectImageAndMaskMetadataTransform( image_ilu.copy(), mask_t, np.concatenate( ( [label], h),axis=0 ) )
                        
        else: 
            assert(False)         

        if self.transform_image: 
            obj_image = self.transform_image( obj_image ) 

        if self.transform_data: 
            obj_data = self.transform_data( obj_data )
            
        x_img, y_mask, y_lab = obj_data.to_value()
        x_org = obj_image.to_value()
        
        return x_org, x_img, y_mask, y_lab


