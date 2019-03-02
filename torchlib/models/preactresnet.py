import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import random


__all__ = ['PreActResNet', 'PreActResEmbNet', 'PreActResEmbExNet', 'preactresnet18', 'preactresnet34', 
            'preactresnet50', 'preactresnet101','preactresnet152', 'preactresembnet18', 'preactresembnetex18', 
            'preactresembnetex34']

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
    def __init__(self, block, num_blocks,  num_classes=1000, num_channels=3, initial_channels=64):
        super(PreActResNet, self).__init__()
        self.in_planes = initial_channels
        self.num_classes = num_classes
        self.num_channels=num_channels
        self.size_input=32

        self.conv1 = nn.Conv2d(num_channels, initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, initial_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, initial_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, initial_channels*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(initial_channels*8*block.expansion, num_classes)

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
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        #out = F.avg_pool2d(out, out.shape[3] )
        out = F.adaptive_avg_pool2d( out, 1 )
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def preactresnet18( pretrained=False, **kwargs ):    
    model = PreActResNet(PreActBlock, [2,2,2,2], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pass
    return model

def preactresnet34(pretrained=False, **kwargs):
    model =  PreActResNet(PreActBlock, [3,4,6,3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pass
    return model

def preactresnet50(pretrained=False, **kwargs):
    model =  PreActResNet(PreActBottleneck, [3,4,6,3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pass
    return model

def preactresnet101(pretrained=False, **kwargs):
    model =  PreActResNet(PreActBottleneck, [3,4,23,3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pass
    return model

def preactresnet152(pretrained=False, **kwargs):
    model =  PreActResNet(PreActBottleneck, [3,8,36,3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pass
    return model



class PreActResEmbNet(nn.Module):
    def __init__(self, block, num_blocks,  dim=64, num_channels=3, initial_channels=64):
        super(PreActResEmbNet, self).__init__()
        self.in_planes = initial_channels
        self.dim = dim
        self.num_channels=num_channels
        self.size_input=32 
        self.conv_dim_out = initial_channels*8*block.expansion #ex: 64*8*4

        self.conv1 = nn.Conv2d(num_channels, initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, initial_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, initial_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, initial_channels*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(self.conv_dim_out , dim)
        

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
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = F.avg_pool2d(out, out.shape[3] )   
        out = F.adaptive_avg_pool2d( out, 1 )

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def preactresembnet18( pretrained=False, **kwargs ):    
    model = PreActResEmbNet(PreActBlock, [2,2,2,2], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pass
    return model


class PreActResEmbExNet(nn.Module):
    def __init__(self, block, num_blocks, dim=64, num_classes=1000, num_channels=3, initial_channels=64):
        super(PreActResEmbExNet, self).__init__()
        self.in_planes = initial_channels
        self.dim = dim
        self.num_classes=num_classes
        self.num_channels=num_channels
        self.size_input=32 
        self.conv_dim_out = initial_channels*8*block.expansion*1 #ex: 64*8*4

        self.conv1 = nn.Conv2d(num_channels, initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, initial_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, initial_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, initial_channels*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(self.conv_dim_out , dim)
        self.classification = nn.Linear(dim , num_classes)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight.data)    
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x ):
        
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
               
        #out = F.avg_pool2d(out, out.shape[3] )
        #out = F.avg_pool2d(out, 4 )  
        out = F.adaptive_avg_pool2d( out, 1 )
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        y = self.classification(out)      
        
        return out, y

    
def preactresembnetex18( pretrained=False, **kwargs ):    
    model = PreActResEmbExNet(PreActBlock, [2,2,2,2], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pass
    return model    


def preactresembnetex34( pretrained=False, **kwargs ):    
    model = PreActResEmbExNet(PreActBlock, [3,4,6,3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pass
    return model    

