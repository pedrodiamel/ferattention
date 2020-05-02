
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 12 * 12, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2 )
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        
        #print(xs.shape)
        #assert(False)
                
        xs = xs.view(-1, 10 * 12 * 12)
        xs = self.fc_loc(xs)
        theta = xs.view(-1, 2, 3)
                
        #grid = F.affine_grid(theta, x.size())
        #x = F.grid_sample(x, grid)

        return theta

    def forward(self, x):
        # transform the input
        theta = self.stn(x)        
        return theta

    
    
    
def test():
    num_channels=1
    num_classes=10
    net = STN()
    y = net( torch.randn(1,num_channels,64,64) )
    print(y.size())
    
# test()
    
    
    
    
    
    
    
