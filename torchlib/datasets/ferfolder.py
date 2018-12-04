
import os
import numpy as np
import pandas as pd
from pytvision.datasets.imageutl import dataProvide


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(pathdir, extensions):
    images = []
    targets = []          
    for root, _, fnames in sorted(os.walk(pathdir)):        
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname) 
                target = (fname.split('.')[0].split('_')[1])    
                target = int(target)                
                item = (path, target)
                images.append(item)
                targets.append(target)
    return images, np.array(targets)





class FERFolderDataset( dataProvide ):
    r"""FER Folder dataset
    """

    def __init__(self, 
        pathname, 
        train=True,
        idenselect=[],
        shuffle=False,
        transform=None,
        download=False,
        extensions=IMG_EXTENSIONS,
        ):

        self.pathname  = os.path.expanduser( pathname )
        self.shuffle   = shuffle
        self.transform = transform   
        self.data, self.labels = make_dataset( pathname, extensions )
          
            
        df = pd.read_csv( os.path.join(pathname, 'ids.csv' ) )
        self.iactor = df.as_matrix()[:,0]
        
        #idenselect = np.arange(20)
        indexs = np.ones( (len(self.data) ,1) )
        actors = np.unique( self.iactor )
        for i in idenselect:
            indexs[self.iactor == actors[i]] = 0       
        self.indexs = np.where(indexs == train)[0] 
        
        self.labels = self.labels[ self.indexs ]
        self.classes = np.unique( self.labels )
        self.numclass = len(self.classes)  
                
        

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, i):   
            
        #check index
        if i<0 and i>len(self.indexs): 
            raise ValueError('Index outside range') 
        
        idx = self.indexs[i]        
        self.index = idx
        pathname   = self.data[idx][0]
        label      = self.data[idx][1]
        image      = np.array(self._loadimage(pathname), dtype=np.uint8)

        return image, label


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        return fmt_str