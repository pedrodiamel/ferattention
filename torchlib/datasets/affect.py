import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from pytvision.datasets.imageutl import dataProvide

def make_dataset( path, metadata, train, org ):
    '''load file patch for disk
    '''
    
    #print( os.path.join( path, metadata))
    data = pd.read_csv( os.path.join( path, metadata) )   
    # filter dataset for only expression and valid image data
    #data = data[ [ exp in [0,1,2,3,4,5,6,7] for i,exp in enumerate(data['expression']) if (i != 235929 or i != 315313) ]   ] 
    ifilter = np.array([ exp in [0,1,2,3,4,5,6,7] for i,exp in enumerate( data['expression'] )  ])    
    
    if train:
        ifilter[ np.array( [235929, 315313] if org else [235929, 315313, 126295]   )  ] = False
    
    
    ifilter = np.where( ifilter == True )[0]
    return data, ifilter


class AffectNetProvide( dataProvide ):
    '''Provide for AffectNet dataset
    
    Format
    subDirectory_filePath;face_x;face_y;face_width;face_height;facial_landmarks;expression;valence;arousal
    '''
            
    classes = ['Neutral - NE', 'Happiness - HA', 'Surprise - SU', 'Sadness - SA', 'Anger - AN', 'Disgust - DI', 'Fear - FR', 'Contempt - CO']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}
    emo2ck = [0,1,3,2,6,5,4,7]
    
    @classmethod
    def create(
        cls, 
        path,
        train=True,
        folders_images='train',
        metadata='train.csv',
        org=True,
        ):
        '''
        Factory function that create an instance of ATLASProvide and load the data form disk.
        '''
        provide = cls(path, train, folders_images, metadata, org )
        return provide
    
    def __init__(self,
        path,        
        train=True,
        folders_images='train',
        metadata='train.csv',
        org=True,
        ):
        super(AffectNetProvide, self).__init__( )        
        self.path            = os.path.expanduser( path )
        self.folders_images  = folders_images
        self.metadata        = metadata
        self.data            = []
        self.train           = train
        self.org             = org 
        
        self.data, self.indexs = make_dataset( self.path, self.metadata, self.train, self.org  )
        self.labels = np.array([ self.emo2ck[ self.data['expression'][ self.indexs[i] ] ] for i in range(len(self.indexs))  ])
        self.classes = np.unique( self.labels )
        self.numclass = len(self.classes)        
        
        
                
    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, i):          
        #check index
        if i<0 and i>len(self.indexs): raise ValueError('Index outside range')
        j = self.indexs[i]
        self.index = j              
        pathimagefile = self.data['subDirectory_filePath'][j]
        label = self.labels[i]        
        image = np.array(self._loadimage( os.path.join(self.path, self.folders_images, pathimagefile if self.org else pathimagefile.split('.')[0] + '.png' ) ), dtype=np.uint8 )
        return image, label

    
    def getpathname(self):
        return self.data['subDirectory_filePath'][ self.index ]
    
    def getlandamarks(self):
        i = self.index
        return np.array([  float(x)   for x in self.data['facial_landmarks'][i].split(';') ]).reshape( -1, 2 )

    def bbox(self):
        i = self.index
        return (
            self.data['face_x'][i], 
            self.data['face_y'][i],
            self.data['face_width'][i],
            self.data['face_height'][i]
            )
    

def create_affect( path, train=True ):    
    folders_images='Manually_Annotated/Manually_Annotated_Images'
    metadata = 'training.csv'  if train else 'validation.csv' 
    return AffectNetProvide.create(
        path=path, 
        train=train, 
        folders_images=folders_images, 
        metadata=metadata 
    )

def create_affectdark( path, train ):
    folders_images='affectnetdarck'
    metadata = 'training.csv' if train else 'validation.csv' 
    return AffectNetProvide.create(
        path=path, 
        train=train, 
        folders_images=folders_images, 
        metadata=metadata,
        org=False
    )    

