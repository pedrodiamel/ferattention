#----------------------------------------------------------------------------------------------
# Custom VGG for facial expression recognition
# Paper: Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution
# https://arxiv.org/abs/1608.01041
# Pedro D. Marrero Fernandez
#----------------------------------------------------------------------------------------------

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

__all__ = ['CVGG13', 'cvgg13', 'CVGGEmb13', 'cvggemb13', 'CVGGEmbEx13', 'cvggembex13']

def cvgg13(pretrained=False, **kwargs):
    r"""CVGG13 model architecture
    https://arxiv.org/abs/1608.01041
    """
    model = CVGG13(**kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['cvgg13']))
    return model


class CVGG13(nn.Module):
    
    def __init__(self, num_classes=8, num_channels=1, init_weights=True, batch_norm=False):
        super(CVGG13, self).__init__()
                
        self.num_classes=num_classes
        self.num_channels=num_channels
        self.size_input = 64
        self.dim = 256 * 4 * 4 # output layer of representation
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'];
        self.features = self._create_features( cfg, batch_norm=batch_norm, in_channels = num_channels )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def representation(self, x):                
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _create_features(self, cfg, batch_norm=False, in_channels = 1):        
        layers = []        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    

def cvggemb13(pretrained=False, **kwargs):
    r"""CVGGEmb13 model architecture
    https://arxiv.org/abs/1608.01041
    """
    model = CVGGEmb13(**kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['cvgg13']))
    return model


class CVGGEmb13(nn.Module):
    
    def __init__(self, dim=64, num_channels=1, init_weights=True, batch_norm=False):
        super(CVGGEmb13, self).__init__()
        self.dim=dim
        self.num_channels=num_channels
        self.size_input = 64
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'];
        self.features = self._create_features( cfg, batch_norm=batch_norm, in_channels = num_channels )
        
        self.embedded = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, dim),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedded(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _create_features(self, cfg, batch_norm=False, in_channels = 1):        
        layers = []        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    
    
    
def cvggembex13(pretrained=False, **kwargs):
    r"""CVGGEmb13 model architecture
    https://arxiv.org/abs/1608.01041
    """
    model = CVGGEmbEx13(**kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['cvgg13']))
    return model


class CVGGEmbEx13(nn.Module):
    
    def __init__(self, dim=64, num_classes=1000, num_channels=1, init_weights=True, batch_norm=False):
        super(CVGGEmbEx13, self).__init__()
        self.dim=dim
        self.num_channels=num_channels
        self.size_input = 64
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'];
        self.features = self._create_features( cfg, batch_norm=batch_norm, in_channels = num_channels )
        
        self.embedded = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, dim),
        )
        self.classification = nn.Linear(dim , num_classes)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedded(x)
        y = self.classification(x)
        return x, y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _create_features(self, cfg, batch_norm=False, in_channels = 1):        
        layers = []        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
