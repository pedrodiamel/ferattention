from torchlib import models as md
import torch.nn as nn
import torch
import unittest
import sys
import os

sys.path.append('../')


class TestModel(unittest.TestCase):

    def test_model_resnet(self):
        num_channels = 1
        num_classes = 10
        imsize = 64
        net = md.resnet18(False, num_classes=num_classes,
                          num_channels=num_channels)
        y = net(torch.randn(1, num_channels, imsize, imsize))
        self.assertEqual(y.shape[1], num_classes)

    def test_model_preactresnetembcls(self):
        num_channels = 1
        num_classes = 10
        dim = 10
        imsize = 128
        net = md.preactresembnetex18(
            False, dim=dim, num_channels=num_channels, num_classes=num_classes)
        xemb, y = net(torch.randn(1, num_channels, imsize, imsize))
        self.assertEqual(xemb.shape[1], dim)
        self.assertEqual(y.shape[1], num_classes)

    def test_model_ferattention(self):
        num_channels = 3
        num_classes = 10
        dim = 32
        imsize = 64

        for backbone in ['preactresnet', 'inception', 'resnet', 'cvgg']:
            net = md.ferattention(False, dim=dim, num_classes=num_classes,  num_channels=num_channels, backbone=backbone)
            y, att, g_att, g_ft = net(torch.randn(
                1, num_channels, imsize, imsize))
            self.assertEqual(y.shape[1], num_classes)
            self.assertEqual(att.shape[1], num_channels)
            self.assertEqual(att.shape[2], imsize)
            self.assertEqual(att.shape[3], imsize)
            self.assertEqual(g_att.shape[1], num_classes)
            self.assertEqual(g_att.shape[2], imsize)
            self.assertEqual(g_att.shape[3], imsize)
            self.assertEqual(g_ft.shape[1], num_classes)
            self.assertEqual(g_ft.shape[2], imsize)
            self.assertEqual(g_ft.shape[3], imsize)

    def test_model_ferattentiongmm( self ):
        num_channels=3
        num_classes=10
        dim=32
        imsize=64

        #TODO: Create backbone for 'inception'
        for backbone in ['preactresnet', 'resnet', 'cvgg']:
            net = md.ferattentiongmm( False, dim=dim, num_classes=num_classes, num_channels=num_channels, backbone=backbone)
            z, y, att, g_att, g_ft = net(  torch.randn(1, num_channels, imsize, imsize ) )

            self.assertEqual( z.shape[1], dim )
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


    def test_model_ferattentionstn(self):
        num_channels = 3
        num_classes = 10
        dim = 32
        imsize = 64

        for backbone in ['preactresnet', 'inception', 'resnet', 'cvgg']:
            net = md.ferattentionstn(False, dim=dim, num_classes=num_classes,  num_channels=num_channels, backbone=backbone)
            y, att, theta, att_t, g_att, g_ft = net(torch.randn(
                1, num_channels, imsize, imsize))

            self.assertEqual(y.shape[1], num_classes)
            self.assertEqual(att.shape[1], num_channels)
            self.assertEqual(att.shape[2], imsize)
            self.assertEqual(att.shape[3], imsize)
            self.assertEqual(g_att.shape[1], num_classes)
            self.assertEqual(g_att.shape[2], imsize)
            self.assertEqual(g_att.shape[3], imsize)
            self.assertEqual(g_ft.shape[1], num_classes)
            self.assertEqual(g_ft.shape[2], imsize)
            self.assertEqual(g_ft.shape[3], imsize)
            self.assertEqual(att_t.shape[1], num_channels)
            self.assertEqual(att_t.shape[2], imsize)
            self.assertEqual(att_t.shape[3], imsize)


    def test_model_ferattentiongmmstn( self ):
        num_channels=3
        num_classes=10
        dim=32
        imsize=64

        #TODO: Create backbone for 'inception'
        for backbone in ['preactresnet', 'resnet', 'cvgg']:
            net = md.ferattentiongmmstn( False, dim=dim, num_classes=num_classes, num_channels=num_channels, backbone=backbone)
            z, y, att, theta, att_t, g_att, g_ft = net(  torch.randn(1, num_channels, imsize, imsize ) )

            self.assertEqual( z.shape[1], dim )
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
            self.assertEqual(att_t.shape[1], num_channels)
            self.assertEqual(att_t.shape[2], imsize)
            self.assertEqual(att_t.shape[3], imsize)


if __name__ == '__main__':
    unittest.main()
