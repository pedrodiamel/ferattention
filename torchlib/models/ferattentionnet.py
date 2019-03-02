
import random

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

from . import preactresnet
from . import stn


__all__ = ['FERAttentionNet', 'FERAttentionSTNNet', 'ferattention', 'ferattentionstn' ]



def ferattention(pretrained=False, **kwargs):
    """"FERAttention model architecture
    """
    model = FERAttentionNet(pretrained=pretrained, **kwargs)

    if pretrained == True:
        #model.load_state_dict(state['model'])
        pass
    return model

def ferattentionstn(pretrained=False, **kwargs):
    """"FERAttentionSTN model architecture
    """
    model = FERAttentionSTNNet(pretrained=pretrained, **kwargs)

    if pretrained == True:
        #model.load_state_dict(state['model'])
        pass
    return model

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
        self.cr1 = ConvRelu(_in, _out)
        self.cr2 = ConvRelu(_in, _out)
    def forward(self, x):
        x = self.cr1(x)
        x = self.cr1(x)
        return x

class Coder(nn.Module):
    def __init__(self, in_size, out_size ):
        super(Coder, self).__init__()
        self.conv = ConvRelu2(in_size, out_size )
        self.down = nn.MaxPool2d(2, 2)
    def forward(self, x):
        y1  = self.conv(x)
        y2  = self.down(y1)
        return y1, y2

class Decoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Decoder, self).__init__()
        self.conv = ConvRelu2(in_size, out_size)
        self.up = nn.functional.interpolate
    def forward(self, x1, x2):
        x2 = self.up(x2, scale_factor=2  ,mode='bilinear', align_corners=False)
        return self.conv( torch.cat([x1, x2], 1))  

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels ):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels
        self.up = nn.functional.interpolate
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
    
    def __init__(self, in_channels=3, out_channels=1, bbrelu=False):
        super(AttentionNet, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.bbrelu = bbrelu    

        filters = [64, 128, 256]
        self.down1  = Coder(     in_channels,           filters[0])
        self.down2  = Coder(     filters[0],            filters[1])
        self.center = ConvRelu2( filters[1],            filters[2])
        self.up2    = Decoder(   filters[2]+filters[1], filters[1])
        self.up1    = Decoder(   filters[1]+filters[0], filters[0])
        self.final  = nn.Conv2d( filters[0],            out_channels, 1)
        self.brelu  = BiReLU()


    def forward(self, x):        
        x,befdown1 = self.down1(x)
        x,befdown2 = self.down2(x)
        x = self.center(x)
        x = self.up2(befdown2, x)
        x = self.up1(befdown1, x)
        x = self.final(x)

        if self.bbrelu:
            x = self.brelu(x)
            #x = x * ( torch.abs(x) > 0.02 ).float()

        return x

class AttentionResNet(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, bbrelu=False, num_filters=32, encoder_depth=34, pretrained=True):
        super(AttentionNet, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.bbrelu = bbrelu    
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

        self.center = DecoderBlockV2(bottom_channel_nr,                        num_filters * 8 * 2, num_filters * 8 )              
        self.dec5   = DecoderBlockV2(bottom_channel_nr + num_filters * 8,      num_filters * 8 * 2, num_filters * 8)
        self.dec4   = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3   = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2   = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2)
        self.dec1   = DecoderBlockV2(num_filters * 2 * 2,                      num_filters * 2 * 2, num_filters)
        
        self.attention_map = nn.Sequential(
            ConvRelu(num_filters, num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1)  
        )  

        self.brelu  = BiReLU()


    def forward(self, x):        

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
        x = self.attention_map( dec1 ) 

        if self.bbrelu:
            x = self.brelu(x)
            #x = x * ( torch.abs(x) > 0.02 ).float()

        return x

class FERAttentionNet(nn.Module):
    """FERAttentionNet
    """
    
    def __init__(self, encoder_depth, dim=32, num_classes=1, num_channels=3, num_filters=32 ):
        
        super().__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters

        #attention module
        # TODO March 01, 2019: Include select model attention
        self.attention_map = AttentionNet( in_channels=num_channels, out_channels=1  )             
        
        #feature module
        self.conv_input = nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=9, stride=1, padding=4, bias=True)
        self.feature    = self.make_layer(_Residual_Block_SR, 4, num_filters )
        self.conv_mid   = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        
        #recostruction
        self.reconstruction = nn.Sequential(
            ConvRelu(num_filters, num_filters//2),
            ConvRelu(num_filters//2, num_filters//4),
            nn.Conv2d(in_channels=num_filters//4, out_channels=num_channels, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.LeakyReLU(0.2, inplace=True),
        )

        #classification and reconstruction
        # TODO March 01, 2019: Select of classification and representation module 
        self.netclass = preactresnet.preactresembnetex18( num_classes=num_classes, dim=dim, num_channels=num_channels  )
    
    def make_layer(self, block, num_of_layer, num_ft):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_ft))
        return nn.Sequential(*layers)
        

    def forward(self, x, x_org=None ):
                 
        
        #attention map
        g_att = self.attention_map( x ) 
            
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
           
        
        att_out = att      
        # if self.training:
        #     att_out = att          
        #     if random.random() < 0.50:
        #         if random.random() < 0.25:
        #             att_out = x_org
        #         else: 
        #             att_out = att
        
        
        #classification
        att_pool = F.avg_pool2d(att_out, 2) # <- 32x32 source                     
        z, y = self.netclass( att_pool )
  

        # #ensamble classification
        # #x = x * ( torch.abs(att) <= 0.02 ).float()
        # out = [ att_t,  att  ] #x
        # z=[]; y=[]
        # for o in out:
        #     att_pool = F.avg_pool2d(o, 2) # <- 32x32 source 
        #     zs, ys = self.netclass( att_pool )
        #     z.append(zs)
        #     y.append(ys)            
        # z = torch.stack(z, dim=2).mean(dim=2)
        # y = torch.stack(y, dim=2).mean(dim=2)
                  
            
        return z, y, att, g_att, g_ft 
 
class FERAttentionSTNNet(nn.Module):
    """FERAttentionSTNNet
    """
    
    def __init__(self, encoder_depth, dim=32, num_classes=1, num_channels=3, num_filters=32 ):
        
        super().__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters

        #attention module
        # TODO March 01, 2019: Include select model attention
        self.attention_map = AttentionNet( in_channels=num_channels, out_channels=1  )             
        
        #feature module
        self.conv_input = nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=9, stride=1, padding=4, bias=True)
        self.feature    = self.make_layer(_Residual_Block_SR, 4, num_filters )
        self.conv_mid   = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        
        #recostruction
        self.reconstruction = nn.Sequential(
            ConvRelu(num_filters, num_filters//2),
            ConvRelu(num_filters//2, num_filters//4),
            nn.Conv2d(in_channels=num_filters//4, out_channels=num_channels, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.LeakyReLU(0.2, inplace=True),
        )
        
        #stn
        self.stn = stn.STN()

        #classification and reconstruction
        # TODO March 01, 2019: Select of classification and representation module 
        self.netclass = preactresnet.preactresembnetex18( num_classes=num_classes, dim=dim, num_channels=num_channels  )
   
    def make_layer(self, block, num_of_layer, num_ft):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_ft))
        return nn.Sequential(*layers)
  
    def forward(self, x, x_org=None ):
                 
        
        #attention map
        g_att = self.attention_map( x ) 
            
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
    
        
        #stn
        theta = self.stn( att.mean(dim=1).unsqueeze(dim=1).detach() ) 
        grid = F.affine_grid(theta, att.size())
        att_t = F.grid_sample(att, grid)   
        
        
        att_out = att_t        
        # if self.training:
        #     att_out = att_t            
        #     if random.random() < 0.50:
        #         if random.random() < 0.25:
        #             att_out = x_org
        #         else: 
        #             att_out = att
        
        
        #classification
        att_pool = F.avg_pool2d(att_out, 2) # <- 32x32 source                     
        z, y = self.netclass( att_pool )
  

        # #ensamble classification
        # #x = x * ( torch.abs(att) <= 0.02 ).float()
        # out = [ att_t,  att  ] #x
        # z=[]; y=[]
        # for o in out:
        #     att_pool = F.avg_pool2d(o, 2) # <- 32x32 source 
        #     zs, ys = self.netclass( att_pool )
        #     z.append(zs)
        #     y.append(ys)            
        # z = torch.stack(z, dim=2).mean(dim=2)
        # y = torch.stack(y, dim=2).mean(dim=2)
                  
            
        return z, y, att, theta, att_t, g_att, g_ft 
    



