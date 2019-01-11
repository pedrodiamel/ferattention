import os
import numpy as np
import h5py

from pytvision.datasets.imageutl import dataProvide


class FERClassicDataset( dataProvide ):
    """
    FER CLASSIC dataset
        CK, JAFFE, BU        
    Args:
        path
        filename
        idenselect
        transform (callable, optional): Optional transform to be applied on a sample.   
    """

    classes = ['Neutral - NE', 'Happiness - HA', 'Surprise - SU', 'Sadness - SA', 'Anger - AN', 'Disgust - DI', 'Fear - FR', 'Contempt - CO']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, 
        path,
        filename,
        idenselect=[],
        train=True,
        transform=None,
        ):
          
        
        if os.path.isdir(path) is not True:
            raise ValueError('Path {} is not directory'.format(path))

        self.path = path
        self.filename = filename
        dir = os.path.join(path, filename + '.mat' )
        f = h5py.File(dir)

        self.data   = np.array(f["data"])
        self.points = np.array(f["points"])
        self.imsize = np.array(f["imsize"])[:,0].astype(int)
        self.iactor = np.array(f["iactor"])[0,:].astype(int) 
        self.labels = np.array(f["iclass"])[0,:].astype(int) - 1
        self.name   = np.array(f["name"])
        self.num    = np.array(f["num"])[0,0].astype(int)
        
        # Emotions class 
        if filename == 'ck' or filename == 'ckp':       
            #classes = ['Neutral - NE', 'Anger - AN', 'Contempt - CO', 'Disgust - DI', 'Fear - FR', 'Happiness - HA', 'Sadness - SA', 'Surprise - SU']
            toferp = [0, 4, 7, 5, 6, 1, 3, 2 ]
        elif filename=='bu3dfe' or filename=='jaffe':
            #classes = ['Neutral - NE', 'Anger - AN', 'Disgust - DI', 'Fear - FR', 'Happiness - HA', 'Sadness - SA', 'Surprise - SU', 'Contempt - CO']
            toferp = [0, 4, 5, 6, 1, 3, 2, 7 ]
        else:
            assert(False)
        
        #self.toferp = toferp
        #self.classes = classes
        #self.class_to_idx = { _class: i for i, _class in enumerate(classes) }    

        self.labels = np.array([ toferp[l] for l in self.labels ])
        self.numclass = len(np.unique(self.labels))

        index = np.ones( (self.num,1) )
        actors = np.unique(self.iactor)
        for i in idenselect:
            index[self.iactor == actors[i]] = 0       
        self.index = np.where(index == train)[0]        
        self.transform = transform
        
        self.labels_org = self.labels
        self.labels = self.labels[ self.index ]
        self.classes = [self.classes[ i ] for i in  np.unique( self.labels ) ]
        self.numclass = len(self.classes)  
              

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):   

        if i<0 and i>len(self.index): raise ValueError('Index outside range')
        i = self.index[i]
        image = np.array( self.data[i].reshape(self.imsize).transpose(1,0), dtype=np.uint8 )
        label = self.labels_org[i]
        return image, label

    def iden(self, i):
        return self.iactor[i]

    def getladmarks(self, i):
        return np.squeeze( self.points[i,...] ).transpose(1,0) * [self.width / self.imsize[0], self.height / self.imsize[1]]
        #return np.squeeze( self.points[i,...]).transpose(1,0)

    def getroi(self):   

        #pts = self.getladmarks()
        #minx = np.min(pts[:,0]); maxx = np.max(pts[:,0]);
        #miny = np.min(pts[:,1]); maxy = np.max(pts[:,1]); 
        #box = [minx,miny,maxx,maxy]
        
        box = [0,0,48,48]
        face_rc = Rect(box)        
        return face_rc
