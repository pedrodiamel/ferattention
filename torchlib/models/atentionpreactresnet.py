
import random

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

from . import preactresnet
from . import stn


__all__ = ['AtentionResNet', 'atentionresnet152', 'atentionresnet101', 'atentionresnet34']



def atentionresnet152(pretrained=False, **kwargs):
    """"AtentionResNet model architecture
    """
    model = AtentionResNet(encoder_depth=152 ,pretrained=pretrained, **kwargs)

    if pretrained == True:
        #model.load_state_dict(state['model'])
        pass
    return model

def atentionresnet101(pretrained=False, **kwargs):
    """"AtentionResNet model architecture
    """
    model = AtentionResNet(encoder_depth=101 ,pretrained=pretrained, **kwargs)

    if pretrained == True:
        #model.load_state_dict(state['model'])
        pass
    return model


def atentionresnet34(pretrained=False, **kwargs):
    """"AtentionResNet model architecture
    """
    model = AtentionResNet(encoder_depth=34 ,pretrained=pretrained, **kwargs)

    if pretrained == True:
#         print('>> pretrained: {} !!!'.format('chk000050') )
#         state = torch.load('../out/fer_atentionresnet34_attgmm_adam_bu3dfe_dim64_preactresnet18x32_fold0_001/models/chk000350.pth.tar')
#         model.load_state_dict( state['state_dict'] )
#         model.netclass.weights_init()     
        pass
    
    return model


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class Conv2D(nn.Module):
    def __init__(self, filtersin, filtersout, kernel_size=(3,3), s=1, pad=0, is_batchnorm=False):
        super(Conv2D, self).__init__()
        if is_batchnorm:
            self.conv = nn.Sequential(nn.Conv2d(filtersin, filtersout, kernel_size, s, pad), nn.BatchNorm2d(filtersout), nn.ReLU(),)
        else:
            self.conv = nn.Sequential(nn.Conv2d(filtersin, filtersout, kernel_size, s, pad), nn.ReLU(),)
    def forward(self, x):
        x = self.conv(x)
        return x
    
class DilateCenter(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, is_batchnorm=False ):
        super(DilateCenter, self).__init__()
        self.in_size = in_size
        self.out_size = out_size  
        self.conv_d1 = nn.Conv2d(in_size,  out_size, kernel_size, 1, kernel_size//2 + 0, dilation=1 )
        self.conv_d2 = nn.Conv2d(out_size, out_size, kernel_size, 1, kernel_size//2 + 1, dilation=2 )
        self.conv_d3 = nn.Conv2d(out_size, out_size, kernel_size, 1, kernel_size//2 + 2, dilation=3 )
        self.conv_d4 = nn.Conv2d(out_size, out_size, kernel_size, 1, kernel_size//2 + 3, dilation=4 )
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()        
    
    def forward(self, x ):         
        skip = torch.zeros( x.shape[0], self.out_size, x.shape[2], x.shape[3] ).cuda()
        x1 = x       
        x2 = self.conv_d1(x1);  skip+= x2
        x3 = self.conv_d2(x2);  skip+= x3 
        x4 = self.conv_d3(x3);  skip+= x4 
        x5 = self.conv_d4(x4);  skip+= x5
        x6 = self.relu (skip)
        y = x6
        return y
    
    
class _Residual_Block_DB(nn.Module):
    def __init__(self, num_ft):
        super(_Residual_Block_DB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
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
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output



class BiReLU(torch.autograd.Function):
    """
    """

    @staticmethod
    def forward(ctx, x):
        """
        """
        ctx.save_for_backward(x)
        alpha=0.01
        return x * ( torch.abs(x) >= alpha ).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        x, = ctx.saved_tensors
        alpha=0.01    
        grad_x = grad_output.clone()
        grad_x[ torch.abs(x) < alpha ] = 0
        return grad_x 


    
class AtentionResNet(nn.Module):
    """PyTorch Atention model using ResNet(34, 101 or 152) encoder.
    """
    
    def __init__(self, encoder_depth, dim=32, num_classes=1, num_channels=3, num_filters=32, dropout_2d=0.2, pretrained=False, is_deconv=True):
        
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        
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
        self.center = DilateCenter( bottom_channel_nr, num_filters * 8 )                
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        
        self.attention_map = nn.Sequential(
            ConvRelu(num_filters, num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1)  
        )                
        
        #feature module
        self.conv_input = nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=9, stride=1, padding=4, bias=True)
        self.feature    = self.make_layer(_Residual_Block_SR, 4, num_filters )
        self.conv_mid   = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        
        #recostruction
        self.reconstruction = nn.Sequential(
            ConvRelu(num_filters, num_filters),
            nn.Conv2d(in_channels=num_filters, out_channels=num_channels, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.stn = stn.STN()

        #classification and reconstruction 
        self.netclass = preactresnet.preactresembnetex18( num_classes=num_classes, dim=dim, num_channels=num_channels  )
        
    
    
    def make_layer(self, block, num_of_layer, num_ft):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_ft))
        return nn.Sequential(*layers)
        

    def forward(self, x, x_org=None ):
                
                
        #attention module
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        center = self.center( conv5 )  
        dec5 = self.dec5(torch.cat([center, conv5], 1))        
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)           
        
        #attention map
        g_att = self.attention_map( dec1 ) 
            
        #feature module
        out = self.conv_input( x )
        residual = out
        out = self.feature( out )
        out = self.conv_mid(out)
        g_ft = torch.add(out, residual )
       
        #fusion
        #\sigma(A) * F(I) 
        attmap = torch.mul( F.sigmoid( g_att ) ,  g_ft )       
        att = self.reconstruction( attmap )   
        #att = BiReLU().apply( att ) 
        #att = att * ( att > 0.1 ).float()
        att = att * ( torch.abs(att) > 0.02 ).float()
        
        
        #stn
        theta = self.stn( att.mean(dim=1).unsqueeze(dim=1).detach() ) 
        grid = F.affine_grid(theta, att.size())
        att_t = F.grid_sample(att, grid)   
        #att_t = att_t * ( att_t >= 0.1 ).float()
        att_t = att_t * ( torch.abs(att_t) >= 0.02 ).float()
        
        
        att_out = att_t        
#         if self.training:
# #             att_out = att            
#             if random.random() < 0.50:
#                 if random.random() < 0.50:
#                     att_out = x_org
#                 else: 
#                     att_out = att
              
        #classification
        att_pool = F.avg_pool2d(att_out, 4) # <- 32x32 source       
        #att_pool = F.dropout(att_pool, training=self.training)        
        z, y = self.netclass( att_pool )
              
        #ensamble classification
#         x = x * ( torch.abs(att) <= 0.02 ).float()
#         out = [ att_t,  att, x ]
#         z=[]; y=[]
#         for o in out:
#             att_pool = F.avg_pool2d(o, 4) # <- 32x32 source 
#             zs, ys = self.netclass( att_pool )
#             z.append(zs)
#             y.append(ys)            
#         z = torch.stack(z, dim=2).mean(dim=2)
#         y = torch.stack(y, dim=2).mean(dim=2)
                  
        return z, y, att, theta, att_t, g_att, g_ft 
    


    
def test():    
    batch=10
    num_channels=3
    num_classes=10
    dim=20    
    net = atentionresnet34( False, dim=dim, num_channels=num_channels, num_classes=num_classes ).cuda()
    z, y, xr = net(  torch.randn(batch, num_channels, 64, 64 ).cuda() ) 
    print( z.shape )
    print( y.shape )
    print( xr.shape )
    
   


if __name__ == "__main__":
    test()


