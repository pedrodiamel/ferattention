
import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd

from .utility import download_url, check_integrity

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


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images



class StanfordOnlineProducts( data.Dataset ):
    
    base_folder = 'Stanford_Online_Products'
    url = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
    filename = 'Stanford_Online_Products.zip'
    zip_md5 = '7f73d41a2f44250d4779881525aea32e'


    def __init__(self, 
            root, 
            train=False, 
            extensions=IMG_EXTENSIONS,
            loader=pil_loader,
            transform=None, 
            target_transform=None, 			
            download=False, 
            ):
        
        
        self.root = os.path.expanduser( root )
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        if download:
            self.download()

        self.root = os.path.join(self.root , self.base_folder )
        
        
        if train:
            data = pd.read_csv(os.path.join(self.root,'Ebay_train.txt') , sep=" ")
            path_train, class_id_train, super_class_id_train = data['path'], data['class_id'], data['super_class_id']
            samples = np.stack( [path_train, class_id_train, super_class_id_train] , axis=1 )
        else:
            data = pd.read_csv(os.path.join(self.root,'Ebay_test.txt') , sep=" ")
            path_test, class_id_test, super_class_id_test = data['path'], data['class_id'], data['super_class_id']
            samples = np.stack( [path_test, class_id_test, super_class_id_test], axis=1 ) 
            
        
        classes = np.array( [ int(s[1]) for s in samples  ] )
        classes_name = np.unique(classes)
        class_to_idx = { classes_name[i]: i for i in range(len(classes_name))}
        samples = np.array( [ (path, class_to_idx[ids], suid ) for path, ids, suid in samples  ] )
        
        classes = np.array( [ int(s[1]) for s in samples  ] )
        classes_name = np.unique(classes)
                
        self.loader = loader
        self.extensions = extensions 
        self.classes = classes_name 
        self.class_to_idx = class_to_idx 
        self.samples = samples 
        self.targets = np.array([int(s[1]) for s in samples])

        self.transform = transform
        self.target_transform = target_transform
        
        
        #self.imgs = [(os.path.join(root, self.base_folder, path), int(class_id) - 1) 
        # for i, (image_id, class_id, super_class_id, path) 
        # in enumerate( map(str.split, open(os.path.join(root, self.base_folder, 'Ebay_{}.txt'.format('train' if train else 'test'))))) 
        # if i > 1]
        
        
    def __getitem__(self, idx):
        """
        Args:
        idx (int): Index

        Returns:
        tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[idx][0], int(self.samples[idx][1])        
        sample = self.loader( os.path.join(self.root, path) )

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


    def _check_integrity(self):
        return os.path.exists( os.path.join(self.root, self.filename))

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.zip_md5)

        # extract file
        cwd = os.getcwd()
        os.chdir(root)
        with zipfile.ZipFile(self.filename, "r") as zip: zip.extractall()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        return fmt_str
    
    
