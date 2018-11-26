import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import h5py
import scipy.io


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



class Cars196( data.Dataset  ):
    
    base_folder_devkit = 'devkit'
    url_devkit = 'http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
    filename_devkit = 'cars_devkit.tgz'
    tgz_md5_devkit = 'c3b158d763b6e2245038c8ad08e45376'

    base_folder_trainims = 'cars_train'
    url_trainims = 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
    filename_trainims = 'cars_ims_train.tgz'
    tgz_md5_trainims = '065e5b463ae28d29e77c1b4b166cfe61'
    
    base_folder_testims = 'cars_test'
    url_testims = 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
    filename_testims = 'cars_ims_test.tgz'
    tgz_md5_testims = '4ce7ebf6a94d07f1952d94dd34c4d501'
    
    url_testanno = 'http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat'
    filename_testanno = 'cars_test_annos_withlabels.mat'
    mat_md5_testanno = 'b0a2b23655a3edd16d84508592a98d10'

    filename_trainanno = 'cars_train_annos.mat'
    base_folder = 'cars_train'	


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

        images = [] 
        classes = []
       
        if train:
            f = scipy.io.loadmat( os.path.join(self.root, self.base_folder_devkit, self.filename_trainanno) )
            base_folder_image = self.base_folder_trainims
        else:
            f = scipy.io.loadmat( os.path.join(self.root, self.filename_testanno) )   
            base_folder_image = self.base_folder_testims
        
        for ant in f['annotations'][0]:
            name = ant[-1][0]
            idn  = int(ant[-2])
            images.append( os.path.join(self.root, base_folder_image ,name)  )            
            classes.append( str(idn) )
        
        classes_name = np.unique(classes)
        class_to_idx = { classes_name[i]: i for i in range(len(classes_name))}
        
        samples=[]
        for i,c in zip(images, classes):
            samples.append((i,class_to_idx[c]))
        
        
        self.extensions = extensions 
        self.classes = classes_name 
        self.class_to_idx = class_to_idx
        self.samples = samples 
        self.targets = np.array([s[1] for s in samples])

        #self.imgs = [(os.path.join(root, self.base_folder_trainims, a[-1][0]), int(a[-2][0]) - 1) 
        # for filename in [self.filename_trainanno] 
        # for a in scipy.io.loadmat(os.path.join(root, self.base_folder_devkit, filename))['annotations'][0] 
        # if (int(a[-2][0]) - 1 < self.num_training_classes) == train] + [(os.path.join(root, self.base_folder_testims, a[-1][0]), int(a[-2][0]) - 1) 
        # for filename in [self.filename_testanno] 
        # for a in scipy.io.loadmat(os.path.join(root, self.base_folder_devkit, filename))['annotations'][0] 
        # if (int(a[-2][0]) - 1 < self.num_training_classes) == train]


    def __getitem__(self, idx):
        """
        Args:
        idx (int): Index

        Returns:
        tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[ idx  ]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)

    def _check_integrity(self):
        return os.path.exists( os.path.join(self.root, self.filename_trainims))


    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        
        root = self.root
    
        urls =[self.url_devkit, self.url_trainims, self.url_testims]
        base_folders = [self.base_folder_devkit, self.base_folder_trainims, self.base_folder_testims]
        filenames = [self.filename_devkit, self.filename_trainims, self.filename_testims]
        tgz_md5s = [self.tgz_md5_devkit, self.tgz_md5_trainims, self.tgz_md5_testims]

        # url_testanno
        # filename_testanno
        # mat_md5_testanno	
    
        for (url, base_folders, filename, tgz_md5) in zip(urls, base_folders, filenames, tgz_md5s):
            download_url(url, root, filename, tgz_md5)
        download_url(self.url_testanno, root, self.filename_testanno, self.mat_md5_testanno)

        # extract file
        cwd = os.getcwd()
        for filename in filenames:
            tar = tarfile.open(os.path.join(root, filename), "r:gz")
            os.chdir(root)
            tar.extractall()
            tar.close()
        os.chdir(cwd)
        
        

        
class Cars196MetricLearning( Cars196 ):
    
    num_training_classes = 98

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

        train_images = [] 
        train_classes = [] 
        
        f = scipy.io.loadmat( os.path.join(self.root, self.base_folder_devkit, self.filename_trainanno) )
        for ant in f['annotations'][0]:
            name = ant[-1][0]
            idn  = int(ant[-2])
            train_images.append( os.path.join(self.root, self.base_folder_trainims ,name)  )            
            train_classes.append( str(idn) )        
         
        
        test_images = [] 
        test_classes = []
        f = scipy.io.loadmat( os.path.join(self.root, self.filename_testanno) )   
        for ant in f['annotations'][0]:
            name = ant[-1][0]
            idn  = int(ant[-2])
            test_images.append( os.path.join(self.root,  self.base_folder_testims  ,name)  )            
            test_classes.append( str(idn) )
        
        
        classes = np.concatenate( (train_classes,test_classes), axis=0 )
        images = np.concatenate( (train_images,test_images), axis=0 )
        
        classes_name = np.unique(classes)
        class_to_idx = { classes_name[i]: i for i in range(len(classes_name))}
        
        samples=[]
        for i,c in zip(images, classes):
            samples.append((i,class_to_idx[c]))
        
        
        self.extensions = extensions 
        self.classes = classes_name 
        self.class_to_idx = class_to_idx
        self.samples = samples 
        self.targets = np.array([s[1] for s in samples])
               
        
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


