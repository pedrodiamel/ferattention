
import os
import sys
import numpy as np
import cv2
import unittest

from pytvision.datasets import imageutl as imutl
from pytvision.transforms import functional as F

sys.path.append('../')
from torchlib.transforms.ferrender import Generator
from torchlib.datasets.factory import FactoryDataset
from torchlib.datasets.fersynthetic import SyntheticFaceDataset

import matplotlib.pyplot as plt


def test_dataset_synthetic():

    data=FactoryDataset.factory(
        pathname='~/.datasets/',
        name=FactoryDataset.ck,
        subset=FactoryDataset.training,
        download=True
        )

    dataset = SyntheticFaceDataset(
        data,
        '~/.datasets/coco',
        generate='image_and_mask',
        count=100,
    )

    img, mask, label = dataset[ np.random.randint( len(dataset) )  ]
    print( len(dataset) )

    plt.figure()
    plt.subplot(121)
    plt.imshow(img.mean(axis=2), cmap='gray' )
    plt.title('Label: {}'.format(label))
    plt.subplot(122)
    plt.imshow( mask[:,:,0] )
    plt.show()


def test_dataset_generator():

    data=FactoryDataset.factory(
        pathname= os.path.expanduser('~/.datasets/'),
        name=FactoryDataset.ck,
        subset=FactoryDataset.training,
        download=True
        )

    ren = Generator()
    img, y = data[ np.random.randint( len(data) ) ]
    img = np.stack((img,img,img),axis=2)

    idx=1
    pathname = os.path.expanduser( '~/.datasets/coco' )
    data_back = imutl.imageProvide( pathname, ext='jpg');
    back = data_back[ (idx)%len(data_back)  ]
    back = F.resize_image(back, 640, 1024, resize_mode='crop', interpolate_mode=cv2.INTER_LINEAR);
    #back = back[:,:,0]
    #back = np.random.randint(255, size=(640, 1024) )


    print(img.shape, img.max())
    print(back.shape, back.max())

    image, image_ilu, mask, h = ren.generate(img, back )
    print( image.shape, image.max() )
    print( mask.shape, mask.max() )

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(image.mean(axis=2), cmap='gray' )
    # plt.subplot(122)
    # plt.imshow( mask[:,:,0] )
    # plt.show()



test_dataset_generator()
# test_dataset_synthetic()
