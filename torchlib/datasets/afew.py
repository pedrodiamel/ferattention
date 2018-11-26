import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import random

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


def make_dataset(dir, folders, subfolder, feature, class_to_idx, extensions ):
    
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(folders):
        
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames): 
                fname = fname.split('.')[0]                
                path = os.path.join( dir, subfolder, feature,  fname )

                paths = []
                for root, _, frames in sorted(os.walk(path)):
                    for frame in sorted(frames):
                        if has_file_allowed_extension(frame, extensions):
                            path = os.path.join( root, frame )
                            paths.append( path )
                paths=paths[4:-4]
                if len(paths) != 0:
                    item = (paths, len(paths), class_to_idx[target])
                    images.append(item)
    return images


class Afew( data.Dataset  ):

    base_folder_train = 'Train_AFEW'
    base_folder_val = 'Val_AFEW'
    base_folder_test = 'Test_AFEW'   

    metadata_folder = 'AlignedFaces_LBPTOP_Points'
    metadata_folder_val = 'AlignedFaces_LBPTOP_Points_Val'
    
    face_folder='Faces'
    LBPTOP_folder='LBPTOP'
    points_folder='Points'

    
    classes = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}


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
        self.extensions = extensions 


        # if download:
        #     self.download()

        base_folder = self.base_folder_train if train else self.base_folder_val
        metadata_folder = self.metadata_folder if train else self.metadata_folder_val

        samples = make_dataset( 
            os.path.join(self.root, base_folder), 
            self.classes, 
            metadata_folder,
            self.face_folder,
            self.class_to_idx,
            extensions, )


        self.samples = samples
        self.targets = [s[-1] for s in samples]



    def __getitem__(self, idx):
        """
        Args:
        idx (int): Index

        Returns:
        tuple: (sample, target) where target is class_index of the target class.
        """
        paths, l, target = self.samples[  idx  ]
        i = random.randint(0,l-1)

        #print(paths)
        #print(len(paths), i, l, target)

        sample = self.loader(paths[i])

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def get_frame(self, idx, iframe ):
        paths, l, target = self.samples[  idx  ]
        if iframe > l: assert(False)

        sample = self.loader(paths[ iframe ])
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

        

    def __len__(self):
        return len(self.samples)