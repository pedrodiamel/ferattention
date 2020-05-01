
import random

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

from . import utils as utl
from . import preactresnet
from . import resnet
from . import inception
from . import cvgg
from . import stn


__all__ = [
           'FERAttentionNet',       'ferattention',
           'FERAttentionSTNNet',    'ferattentionstn',
           'FERAttentionGMMNet',    'ferattentiongmm',
           'FERAttentionGMMSTNNet', 'ferattentiongmmstn'
          ]


def ferattention(pretrained=False, **kwargs):
    """"FERAttention model architecture
    """
    model = FERAttentionNet(**kwargs)
    if pretrained == True:
        #model.load_state_dict(state['model'])
        print('>> pretrained: {} !!!'.format('chk000390') )
        state = torch.load('../chk000390.pth.tar')
        #model.load_state_dict( state['state_dict'] )
        utl.load_state_dict(model.state_dict(), state['state_dict'] )
        model.netclass.weights_init()
        #pass
    return model


def ferattentiongmm(pretrained=False, **kwargs):
    """"FERAttention gmm model architecture
    """
    model = FERAttentionGMMNet(**kwargs)
    if pretrained == True:
        #model.load_state_dict(state['model'])
        print('>> pretrained: {} !!!'.format('chk000100') )
        state = torch.load('../chk000390.pth.tar')
        #model.load_state_dict( state['state_dict'] )
        utl.load_state_dict(model.state_dict(), state['state_dict'] )
        #model.netclass.weights_init()
        #pass
    return model


def ferattentionstn(pretrained=False, **kwargs):
    """"FERAttentionSTN model architecture
    """
    model = FERAttentionSTNNet(**kwargs)
    if pretrained == True:
        #model.load_state_dict(state['model'])
        print('>> pretrained: {} !!!'.format('chk000390') )
        state = torch.load('../chk000390.pth.tar')
        #model.load_state_dict( state['state_dict'] )
        utl.load_state_dict(model.state_dict(), state['state_dict'] )
        model.netclass.weights_init()
        #pass
    return model



def ferattentiongmmstn(pretrained=False, **kwargs):
    """"FERAttentionSTN model architecture
    """
    model = FERAttentionGMMSTNNet(**kwargs)
    if pretrained == True:
        #model.load_state_dict(state['model'])
        print('>> pretrained: {} !!!'.format('chk000390') )
        state = torch.load('../chk000390.pth.tar')
        #model.load_state_dict( state['state_dict'] )
        utl.load_state_dict(model.state_dict(), state['state_dict'] )
        model.netclass.weights_init()
        #pass
    return model



def normalize_layer(x):
    x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.2023 / 0.5) + (0.4914 - 0.5) / 0.5
    x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.1994 / 0.5) + (0.4822 - 0.5) / 0.5
    x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.2010 / 0.5) + (0.4465 - 0.5) / 0.5
    x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
    return x

def conv3x3(_in, _out):
    return nn.Conv2d(_in, _out, kernel_size=3, stride=1, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, _in, _out):
        super().__init__()
        self.conv = conv3x3(_in,_out)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class ConvRelu2(nn.Module):
    def __init__(self, _in, _out):
        super(ConvRelu2, self).__init__()
        self.cr1 = ConvRelu(_in , _out)
        self.cr2 = ConvRelu(_out, _out)
    def forward(self, x):
        x = self.cr1(x)
        x = self.cr2(x)
        return x

class Coder(nn.Module):
    def __init__(self, in_size, out_size ):
        super(Coder, self).__init__()
        self.conv = ConvRelu2( in_size, out_size )
        self.down = nn.MaxPool2d(2,2)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = self.down(y1)
        return y2, y1

class Decoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Decoder, self).__init__()
        self.conv = ConvRelu2(in_size, out_size)
        self.up   = F.interpolate
    def forward(self, x1, x2):
        x2 = self.up(x2, scale_factor=2  ,mode='bilinear', align_corners=False)
        return self.conv( torch.cat([x1, x2], 1) )

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels ):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels
        self.up  = F.interpolate
        self.cr1 = ConvRelu(in_channels,     middle_channels)
        self.cr2 = ConvRelu(middle_channels, out_channels)
    def forward(self, x):
        x = self.up(x, scale_factor=2 ,mode='bilinear', align_corners=False)
        x = self.cr2( self.cr1(x) )
        return x

class _Residual_Block_DB(nn.Module):
    def __init__(self, num_ft):
        super(_Residual_Block_DB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output

class _Residual_Block_SR(nn.Module):
    def __init__(self, num_ft):
        super(_Residual_Block_SR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu  = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output

class BiReLU(torch.autograd.Function):
    """BiReLU
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        alpha=0.01
        return x * ( torch.abs(x) >= alpha ).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha=0.01
        grad_x = grad_output.clone()
        grad_x[ torch.abs(x) < alpha ] = 0
        return grad_x


class AttentionNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super(AttentionNet, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        filters = [64, 128, 256]

        self.down1  = Coder(     in_channels,           filters[0])
        self.down2  = Coder(     filters[0],            filters[1])
        self.center = ConvRelu2( filters[1],            filters[2])
        self.up2    = Decoder(   filters[2]+filters[1], filters[1])
        self.up1    = Decoder(   filters[1]+filters[0], filters[0])
        self.final  = nn.Conv2d( filters[0],            out_channels, 1)

    def forward(self, x):
        x,befdown1 = self.down1(x)
        x,befdown2 = self.down2(x)
        x = self.center(x)
        x = self.up2(befdown2, x)
        x = self.up1(befdown1, x)
        x = self.final(x)
        return x


class AttentionResNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, num_filters=32, encoder_depth=34, pretrained=True):
        super(AttentionResNet, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        #attention module
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr,                        num_filters * 8 * 2, num_filters * 8)
        self.dec5   = DecoderBlockV2(bottom_channel_nr + num_filters * 8,      num_filters * 8 * 2, num_filters * 8)
        self.dec4   = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3   = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2   = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2)
        self.dec1   = DecoderBlockV2(num_filters * 2 * 2,                      num_filters * 2 * 2, num_filters)

        self.attention_map = nn.Sequential(
            ConvRelu(num_filters, num_filters),
            nn.Conv2d(num_filters, out_channels, kernel_size=1)
        )



    def forward(self, x):

        #attention module
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        pool = self.pool(conv5)
        center = self.center( pool )
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)

        #attention map
        x = self.attention_map( dec1 )
        return x


class FERAttentionNet(nn.Module):
    """FERAttentionNet
    """

    def __init__(self, dim=32, num_classes=1, num_channels=3, backbone = 'resnet', num_filters=32 ):

        super().__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters

        # Attention module
        # TODO March 01, 2019: Include select backbone model attention
        self.attention_map = AttentionResNet( in_channels=num_channels, out_channels=num_classes, pretrained=True  )

        # Feature module
        self.conv_input = nn.Conv2d(in_channels=num_channels, out_channels=num_classes, kernel_size=9, stride=1, padding=4, bias=True)
        self.feature    = self.make_layer(_Residual_Block_SR, 8, num_classes )
        self.conv_mid   = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=1, bias=True)

        # Recostruction module
        self.reconstruction = nn.Sequential(
            ConvRelu( 2*num_classes+num_channels, num_filters),
            nn.Conv2d(in_channels=num_filters, out_channels=num_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.conv2_bn = nn.BatchNorm2d(num_channels)


        # Select backbone classification and reconstruction
        # TODO March 01, 2019: Select backbone for classification and representation module
        self.backbone = backbone
        if   self.backbone == 'preactresnet':
             self.netclass = preactresnet.preactresnet18( num_classes=num_classes, num_channels=num_channels )
        elif self.backbone == 'inception':
             self.netclass = inception.inception_v3( num_classes=num_classes, num_channels=num_channels, transform_input=False, pretrained=True )
        elif self.backbone == 'resnet':
             self.netclass = resnet.resnet18( num_classes=num_classes, num_channels=num_channels )
        elif self.backbone == 'cvgg':
             self.netclass = cvgg.cvgg13( num_classes=num_classes, num_channels=num_channels )
        else:
            assert(False)



    def make_layer(self, block, num_of_layer, num_ft):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_ft))
        return nn.Sequential(*layers)


    def forward(self, x, x_org=None ):

        # Attention map
        g_att = self.attention_map( x )

        # Feature module
        out = self.conv_input( x )
        residual = out
        out = self.feature( out )
        out = self.conv_mid(out)
        g_ft = torch.add( out, residual )

        # Fusion
        # \sigma(A) * F(I)
        attmap = torch.mul( torch.sigmoid( g_att ) ,  g_ft )
        att = self.reconstruction( torch.cat( ( attmap, x, g_att ) , dim=1 ) )
        att = F.relu(self.conv2_bn(att))
        att_out = normalize_layer(att)

        # Select backbone classification
        if self.backbone == 'preactresnet':
            att_pool = F.avg_pool2d(att_out, 2)                                                     #if preactresnet
        elif self.backbone == 'inception':
            att_pool = F.interpolate(att_out, size=(299,299), mode='bilinear', align_corners=False) #if inseption
        elif self.backbone == 'resnet':
            att_pool = F.interpolate(att_out, size=(224,224), mode='bilinear', align_corners=False) #if resnet
        elif self.backbone == 'cvgg':
            att_pool = att_out                                                                       #if vgg
        else:
            assert(False)

        y = self.netclass( att_pool )

        return y, att, g_att, g_ft





class FERAttentionGMMNet(nn.Module):
    """FERAttentionGMMNet
    """

    def __init__(self, dim=32, num_classes=1, num_channels=3, backbone = 'resnet', num_filters=32 ):

        super().__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters

        # Attention module
        # TODO March 01, 2019: Include select backbone model attention
        self.attention_map = AttentionResNet( in_channels=num_channels, out_channels=num_classes, pretrained=True )

        # Feature module
        self.conv_input = nn.Conv2d(in_channels=num_channels, out_channels=num_classes, kernel_size=9, stride=1, padding=4, bias=True)
        self.feature    = self.make_layer(_Residual_Block_SR, 8, num_classes )
        self.conv_mid   = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=1, bias=True)

        # Recostruction module
        self.reconstruction = nn.Sequential(
            ConvRelu( 2*num_classes+num_channels, num_filters),
            nn.Conv2d(in_channels=num_filters, out_channels=num_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.conv2_bn = nn.BatchNorm2d(num_channels)


        # Select backbone for classification and reconstruction
        # TODO March 01, 2019: Select backbone for classification and representation module
        self.backbone = backbone
        if   self.backbone == 'preactresnet':
             self.netclass = preactresnet.preactresembnetex18( dim=dim, num_classes=num_classes, num_channels=num_channels )
        elif self.backbone == 'resnet':
             self.netclass = resnet.resnetembex18(dim=dim, num_classes=num_classes, num_channels=num_channels )
        elif self.backbone == 'cvgg':
             self.netclass = cvgg.cvggembex13(dim=dim, num_classes=num_classes, num_channels=num_channels )
        else:
            assert(False)


    def make_layer(self, block, num_of_layer, num_ft):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_ft))
        return nn.Sequential(*layers)


    def forward(self, x, x_org=None ):

        # Attention map
        g_att = self.attention_map( x )

        # Feature module
        out = self.conv_input( x )
        residual = out
        out = self.feature( out )
        out = self.conv_mid( out )
        g_ft = torch.add( out, residual )

        # Fusion
        # \sigma(A) * F(I)
        attmap = torch.mul( torch.sigmoid( g_att ) ,  g_ft )
        att = self.reconstruction( torch.cat( ( attmap, x, g_att ) , dim=1 ) )
        att = F.relu(self.conv2_bn(att))
        att_out = normalize_layer(att)

        # Select backbone classification
        if   self.backcoder == 'preactresnet':
             att_pool = F.avg_pool2d(att_out, 2)                                                     #if preactresnet
        elif self.backcoder == 'inception':
             att_pool = F.interpolate(att_out, size=(299,299) ,mode='bilinear', align_corners=False) #if inseption
        elif self.backcoder == 'resnet':
             att_pool = F.interpolate(att_out, size=(224,224) ,mode='bilinear', align_corners=False) #if resnet
        elif self.backcoder == 'cvgg':
             att_pool = att_out                                                                       #if vgg
        else:
            assert(False)

        z, y = self.netclass( att_pool )

        return z, y, att, g_att, g_ft

class FERAttentionSTNNet(nn.Module):
    """FERAttentionSTNNet
    """

    def __init__(self, dim=1, num_classes=1, num_channels=3, backbone = 'resnet', num_filters=32 ):

        super().__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters

        # Attention module
        # TODO March 01, 2019: Include select model attention
        self.attention_map = AttentionResNet( in_channels=num_channels, out_channels=num_classes, pretrained=True  )

        # Feature module
        self.conv_input = nn.Conv2d(in_channels=num_channels, out_channels=num_classes, kernel_size=9, stride=1, padding=4, bias=True)
        self.feature    = self.make_layer(_Residual_Block_SR, 8, num_classes )
        self.conv_mid   = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=1, bias=True)

        # Recostruction module
        self.reconstruction = nn.Sequential(
            ConvRelu( 2*num_classes+num_channels, num_filters),
            nn.Conv2d(in_channels=num_filters, out_channels=num_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.conv2_bn = nn.BatchNorm2d(num_channels)

        # Stn module
        self.stn = stn.STN()

        # Select backbone classification and reconstructions
        self.backbone = backbone
        if   self.backbone == 'preactresnet':
             self.netclass = preactresnet.preactresnet18(num_classes=num_classes, num_channels=num_channels )
        elif self.backbone == 'inception':
             self.netclass = inception.inception_v3( num_classes=num_classes, num_channels=num_channels, transform_input=False, pretrained=True )
        elif self.backbone == 'resnet':
             self.netclass = resnet.resnet18( num_classes=num_classes, num_channels=num_channels )
        elif self.backbone == 'cvgg':
             self.netclass = cvgg.cvgg13( num_classes=num_classes, num_channels=num_channels )
        else:
            assert(False)



    def make_layer(self, block, num_of_layer, num_ft):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_ft))
        return nn.Sequential(*layers)

    def forward(self, x, x_org=None ):

        # Attention map
        g_att = self.attention_map( x )

        #feature module
        out = self.conv_input( x )
        residual = out
        out = self.feature( out )
        out = self.conv_mid(out)
        g_ft = torch.add(out, residual )

        # Fusion
        # \sigma(A) * F(I)
        attmap = torch.mul( torch.sigmoid( g_att ) ,  g_ft )
        att = self.reconstruction( torch.cat( ( attmap, x, g_att ) , dim=1 ) )
        att = F.relu(self.conv2_bn(att))


        #stn
        theta = self.stn( att.mean(dim=1).unsqueeze(dim=1).detach() )
        grid = F.affine_grid(theta, att.size())
        att_stn = F.grid_sample(att, grid)
        att_t = normalize_layer(att_stn)


        #Select backbone classification
        if self.backbone == 'preactresnet':
            att_pool = F.avg_pool2d(att_t, 2)                                                     #if preactresnet
        elif self.backbone == 'inception':
            att_pool = F.interpolate(att_t, size=(299,299) ,mode='bilinear', align_corners=False) #if inseption
        elif self.backbone == 'resnet':
            att_pool = F.interpolate(att_t, size=(224,224) ,mode='bilinear', align_corners=False) #if resnet
        elif self.backbone == 'cvgg':
            att_pool = att_t                                                                      #if vgg
        else:
            assert(False)

        # Classification
        y = self.netclass( att_pool )

        return y, att, theta, att_t, g_att, g_ft

class FERAttentionGMMSTNNet(nn.Module):
    """FERAttentionGMMSTNNet
    """

    def __init__(self, dim=32, num_classes=1, num_channels=3, backbone = 'resnet', num_filters=32 ):

        super().__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters

        # Attention module
        # TODO March 01, 2019: select backbone model attention
        self.attention_map = AttentionResNet( in_channels=num_channels, out_channels=num_classes, pretrained=True  )

        # Feature module
        self.conv_input = nn.Conv2d(in_channels=num_channels, out_channels=num_classes, kernel_size=9, stride=1, padding=4, bias=True)
        self.feature    = self.make_layer(_Residual_Block_SR, 8, num_classes )
        self.conv_mid   = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=1, bias=True)

        # Recostruction module
        self.reconstruction = nn.Sequential(
            ConvRelu( 2*num_classes+num_channels, num_filters),
            nn.Conv2d(in_channels=num_filters, out_channels=num_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.conv2_bn = nn.BatchNorm2d(num_channels)

        # Stn module
        self.stn = stn.STN()

        # Classification and reconstruction
        # TODO March 01, 2019: Select backbone model classification
        self.backbone = backbone
        if   self.backbone == 'preactresnet':
             self.netclass = preactresnet.preactresembnetex18(  dim=dim, num_classes=num_classes, num_channels=num_channels )
        elif self.backbone == 'resnet':
             self.netclass = resnet.resnetembex18(dim=dim, num_classes=num_classes, num_channels=num_channels )
        elif self.backbone == 'cvgg':
             self.netclass = cvgg.cvggembex13(dim=dim, num_classes=num_classes, num_channels=num_channels )
        else:
            assert(False)



    def make_layer(self, block, num_of_layer, num_ft):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_ft))
        return nn.Sequential(*layers)

    def forward(self, x, x_org=None ):

        # Attention module
        g_att = self.attention_map( x )

        # Feature module
        out = self.conv_input( x )
        residual = out
        out = self.feature( out )
        out = self.conv_mid(out)
        g_ft = torch.add(out, residual )

        # Fusion
        # \sigma(A) * F(I)
        attmap = torch.mul( torch.sigmoid( g_att ) ,  g_ft )
        att = self.reconstruction( torch.cat( ( attmap, x, g_att ) , dim=1 ) )
        att = F.relu(self.conv2_bn(att))
        att_out = normalize_layer(att)

        # Stn module
        theta = self.stn( att_out.mean(dim=1).unsqueeze(dim=1).detach() )
        grid = F.affine_grid(theta, att_out.size())
        att_t = F.grid_sample(att_out, grid)


        # Select backbone classification
        if   self.backbone == 'preactresnet':
             att_pool = F.avg_pool2d(att_t, 2)                                                     #if preactresnet
        elif self.backbone == 'inception':
             att_pool = F.interpolate(att_t, size=(299,299) ,mode='bilinear', align_corners=False) #if inseption
        elif self.backbone == 'resnet':
             att_pool = F.interpolate(att_t, size=(224,224) ,mode='bilinear', align_corners=False) #if resnet
        elif self.backbone == 'cvgg':
             att_pool = att_out                                                                    #if vgg
        else:
            assert(False)

        # Classification
        z, y = self.netclass( att_pool )

        return z, y, att, theta, att_t, g_att, g_ft
