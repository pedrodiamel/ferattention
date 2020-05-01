import unittest
import sys
import os

sys.path.append('../')

import torch
import torch.nn as nn
from torchlib.datasets import  affect


class TestData( unittest.TestCase ):

    def test_data_affect( self ):

        path="~/.datasets/"
        namedataset='affectnet'
        metadata='training.csv' # train.csv, validation.csv
        folders_images='Manually_Annotated/Manually_Annotated_Images'
        train=True #True, False

        pathname = os.path.join( path,  namedataset )
        dataset = affect.AffectNetProvide.create(path=pathname, train=train, folders_images=folders_images, metadata=metadata)
        image, label = dataset[ 0 ]

        print( len(dataset) )
        print( image.shape )
        print( label )



if __name__ == '__main__':
    unittest.main()
