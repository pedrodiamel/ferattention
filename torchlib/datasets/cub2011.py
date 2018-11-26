import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

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



class CUB2011( data.Dataset ):
    
    base_folder = 'CUB_200_2011'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    train_test_split='train_test_split'
    
    def __init__(self, 
            root, 
            train, 
            extensions=IMG_EXTENSIONS, 
            loader=pil_loader, 
            transform=None, 
            target_transform=None, 
            download=False 
            ):
        
        self.root = os.path.expanduser( root )

        if download:
        	self.download()

        pathimage = os.path.join( self.root,  self.base_folder, 'images' ) 
        classes, class_to_idx = find_classes( pathimage )
        samples = make_dataset(pathimage, class_to_idx, extensions)

        if len(samples) == 0:
            raise RuntimeError( 'Dataset not found or corrupted. You can use download=True to download it' )

        ids, train_index = np.loadtxt( os.path.join(self.root, self.base_folder,'{}.txt'.format( self.train_test_split ) ), unpack=True )           
        self.index = np.where(train_index == train)[0] 

        self.loader = loader
        self.extensions = extensions 
        self.classes = classes 
        self.class_to_idx = class_to_idx 
        self.samples = samples 
        self.targets = np.array([s[1] for s in samples])

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, idx):
        """
        Args:
        idx (int): Index

        Returns:
        tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[  self.index[idx]  ]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.index)

    def _check_integrity(self):
        return os.path.exists( os.path.join(self.root, self.filename))

        # for fentry in (self.train_list + self.test_list):
        # 	filename, md5 = fentry[0], fentry[1]
        # 	fpath = os.path.join(root, self.base_folder, filename)
        # 	if not check_integrity(fpath, md5):
        # 		return False
        # return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        return fmt_str



class CUB2011MetricLearning( CUB2011 ):
    
    num_training_classes = 100

    def __init__(self, 
            root, 
            train,
            extensions=IMG_EXTENSIONS, 
            loader=pil_loader , 
            transform=None, 
            target_transform=None, 
            download=False,
        ):

        super(CUB2011MetricLearning, self).__init__( root, True, transform=transform, target_transform=target_transform, download=download  )
        
        classes = self.classes[:self.num_training_classes] if train else self.classes[self.num_training_classes:(self.num_training_classes+self.num_training_classes)  ]
        
        index = np.array([], dtype=int)
        for c in classes:
            index = np.append(index, np.where( self.targets == self.class_to_idx[c] )[0], axis=0 )           

        class_to_idx = {classes[i]: i for i in range(len(classes))} 
        samples = []
        for i in index:
            path = self.samples[i][0]
            c = self.classes[self.samples[i][1]]
            samples.append( (path, class_to_idx[c])  )


        self.index =  np.array( [ i for i in range(len(index)) ] )
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = np.array([s[1] for s in samples])


        #self.class_to_idx = {class_label: class_label_ind for class_label, class_label_ind in self.class_to_idx.items() if class_label in self.classes}
        #self.imgs = [(image_file_path, class_label_ind) for image_file_path, class_label_ind in self.imgs if class_label_ind in self.class_to_idx.values()]

