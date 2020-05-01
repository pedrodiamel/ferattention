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
        self.assertEqual( y.shape[1], num_classes )

    def test_model_preactresnetembcls( self ):
        num_channels=1
        num_classes=10
        dim=10
        imsize=128
        net = md.preactresembnetex18( False, dim=dim, num_channels=num_channels, num_classes=num_classes )
        xemb, y = net(  torch.randn(1, num_channels, imsize, imsize ) )
        self.assertEqual( xemb.shape[1], dim )
        self.assertEqual( y.shape[1], num_classes )

    def test_model_ferattention( self ):
        num_channels=3
        num_classes=10
        dim=32
        imsize=64
        net = md.ferattention( False, dim=dim, num_classes=num_classes, num_channels=num_channels)
        y, att, g_att, g_ft = net(  torch.randn(1, num_channels, imsize, imsize ) )
        self.assertEqual( y.shape[1], num_classes )
        self.assertEqual( att.shape[1], num_channels )
        self.assertEqual( att.shape[2], imsize )
        self.assertEqual( att.shape[3], imsize )
        self.assertEqual( g_att.shape[1], num_classes )
        self.assertEqual( g_att.shape[2], imsize )
        self.assertEqual( g_att.shape[3], imsize )
        self.assertEqual( g_ft.shape[1], num_classes )
        self.assertEqual( g_ft.shape[2], imsize )
        self.assertEqual( g_ft.shape[3], imsize )


if __name__ == '__main__':
    unittest.main()
