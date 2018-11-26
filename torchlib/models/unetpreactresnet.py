import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import random

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, in_channels, num_classes, block, num_blocks, feature_channels=64):
        super(PreActResNet, self).__init__()
        self.in_planes = feature_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, feature_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, feature_channels,   num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, feature_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, feature_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, feature_channels*8, num_blocks[3], stride=2)

        self.linear = nn.Linear(feature_channels*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x ):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)



def preactresnet18(initial_channels, num_classes):    
    return PreActResNet(initial_channels, num_classes, PreActBlock, [2,2,2,2])

def preactresnet34(initial_channels, num_classes):
    
    return PreActResNet(initial_channels, num_classes, PreActBlock, [3,4,6,3])

def preactresnet50(initial_channels,num_classes):
    return PreActResNet(initial_channels, num_classes, PreActBottleneck, [3,4,6,3] )

def preactresnet101(initial_channels,num_classes):
    return PreActResNet(initial_channels, num_classes, PreActBottleneck, [3,4,23,3] )

def preactresnet152(initial_channels,num_classes):
    return PreActResNet(initial_channels, num_classes, PreActBottleneck, [3,8,36,3])




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
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,  padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class UNetPreActResNet(nn.Module):
    """PyTorch U-Net model using Pre Act ResNet(34, 101 or 152) encoder.
    """

    def __init__(self, encoder_depth, num_classes=1, in_channels=3, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = preactresnet34( in_channels, num_classes  )
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = preactresnet101( in_channels, num_classes  )
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = preactresnet152( in_channels, num_classes  )
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')


        self.pool = nn.MaxPool2d(2, 2)
        #self.relu = nn.ReLU(inplace=True)
        #self.conv1 = self.encoder.conv1

        self.conv1 = nn.Sequential(
            self.encoder.conv1,
            #self.encoder.bn1,
            #self.encoder.relu,
            self.pool
            )

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):         


        print(x.shape)
        conv1 = self.conv1(x);      print(conv1.shape, 'conv1')
        conv2 = self.conv2(conv1);  print(conv2.shape, 'conv2')
        conv3 = self.conv3(conv2);  print(conv3.shape, 'conv3') 
        conv4 = self.conv4(conv3);  print(conv4.shape) 
        conv5 = self.conv5(conv4);  print(conv5.shape)

        pool = self.pool(conv5);                         print(pool.shape) 
        center = self.center(pool);                      print(center.shape, 'center')
          

        dec5 = self.dec5(torch.cat([center, conv5], 1)); print(dec5.shape)     

        
        assert(False)  

        dec4 = self.dec4(torch.cat([dec5, conv4], 1));   print(dec4.shape)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1));   print(dec3.shape)
        dec2 = self.dec2(torch.cat([dec3, conv2], 1));   print(dec2.shape)        
        dec1 = self.dec1(dec2)     

         
        dec0 = self.dec0(dec1)



        y = self.final(F.dropout2d(dec0, p=self.dropout_2d))


        return y






__all__ = ['UNetPreActResNet', 'unetpreactresnet152']

def unetpreactresnet152(pretrained=False, **kwargs):
    """"UNetResNet model architecture
    """
    model = UNetPreActResNet(encoder_depth=34, pretrained=pretrained, **kwargs)
    if pretrained == True:
        #model.load_state_dict(state['model'])
        pass
    return model




def test():
    net = unetpreactresnet152(num_classes=1, in_channels=3 )
    y = net( torch.randn(10,3,64,64) )
    print(y.size())

if __name__ == "__main__":
    test()
# test()

