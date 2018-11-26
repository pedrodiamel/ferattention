
import os
import numpy as np

from pytvision.datasets.imageutl import dataProvide


def make_dataset( path, filename, folder ):
    data = []
    with open( os.path.join(path, filename) , 'r') as file:
        for line in file:
            words = line.split()
            imname, iden = words[:2]
            data.append( ( os.path.join(path, folder, imname), iden )  )
    return data

def read_bbox( path, filename ):
    bboxs=[]
    with open( os.path.join(path, filename) , 'r') as file:
        for i,line in enumerate(file):
            words = line.split()
            if i in [0,1]:
                continue            
            imname, x, y, w, h = words[:5]
            bboxs.append( [int(x), int(y), int(w), int(h)] )
    return bboxs




class CelebaDataset( dataProvide ):
    r"""Celeba dataset
    """

    folder_images  = 'img_align_celeba'
    file_bbox      = 'list_bbox_celeba.txt'
    file_landmarks = 'list_landmarks_align_celeba.txt'
    file_id        = 'identity_CelebA.txt'   


    def __init__(self, 
        pathname, 
        train=True,
        shuffle=False,
        transform=None,
        download=False,
        ):

        self.pathname  = os.path.expanduser( pathname )
        self.shuffle   = shuffle
        self.transform = transform   
        self.data      = make_dataset( self.pathname, self.file_id, self.folder_images )
        #self.bboxs     = read_bbox( self.pathname, self.file_bbox)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):   
        #check index
        if idx<0 and idx>len(self.data): 
            raise ValueError('Index outside range')
        
        self.index = idx
        pathname   = self.data[idx][0]
        image      = np.array(self._loadimage(pathname), dtype=np.uint8)
        image      = image[30:-30,30:-30,:]

        return image


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        return fmt_str