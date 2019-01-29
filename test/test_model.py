import unittest
import sys
import os

sys.path.append('../')

import torch
import torch.nn as nn

from torchlib import models as md

class TestModel( unittest.TestCase ):
    
    def test_model_resnet( self ):  
        num_channels=1
        num_classes=10
        imsize=64
        net = md.resnet18( False, num_classes=num_classes, num_channels=num_channels )
        y = net( torch.randn(1,num_channels, imsize, imsize) )
        print(y.size())
        
    def test_model_preactresnetembcls( self ):
        num_channels=1
        num_classes=10
        dim=10
        imsize=128
        net = md.preactresembnetex18( False, dim=dim, num_channels=num_channels, num_classes=num_classes )
        xemb, y = net(  torch.randn(1, num_channels, imsize, imsize ) ) 
        print( xemb.shape)
        print( y.shape )


if __name__ == '__main__':
    unittest.main()
