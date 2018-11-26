

import numpy as np
from scipy import ndimage


#
# GETWEIGHTMAP: compute the wieght map for data set
#
# @param masks
# @param w0
# @param sigma
#
# @Ref: https://arxiv.org/pdf/1505.04597.pdf
#       https://www.kaggle.com/piotrczapla/tensorflow-u-net-starter-lb-0-34/notebook
#   
# Ecuation 2
#
# w(x) = w_c(x) + w_0*\exp(-\frac{(d_1(x)-d_2(x))^2}{2\sigma^2}) 
#
# where w_c : Ω → R is the weight map to balance the class frequencies, 
# d_1 : Ω → R denotes the distance to the border of the nearest cell and 
# d_2 : Ω → R the distance to the border of the second nearest cell. 
# In our experiments we set w0 = 10 and σ ≈ 5 pixels.
#
#
#

def getunetweightmap( merged_mask, masks, w0=10, sigma=5, ):
    
    # WxHxN to NxWxH
    #masks = masks.transpose( (2,0,1) )
    
    weight = np.zeros(merged_mask.shape)
    # calculate weight for important pixels
    distances = np.array([ndimage.distance_transform_edt(m==0) for m in masks])
    shortest_dist = np.sort(distances, axis=0)
    # distance to the border of the nearest cell 
    d1 = shortest_dist[0]
    # distance to the border of the second nearest cell
    d2 = shortest_dist[1] if len(shortest_dist) > 1 else np.zeros(d1.shape)

    w_b = np.exp(-(d1+d2)**2/(2*sigma**2)).astype(np.float32)
    w_c = getweightmap(merged_mask)
    w = w_c + w0*w_b
    
    #weight = 1 + (merged_mask == 0) * w
    weight = 1 + w
    
    return weight


#weight map
def getweightmap(mask):
    
    w_c = np.empty(mask.shape)
    classes = np.unique(mask)
    frecs = [ np.sum(mask == i)/float(mask.size) for i in classes ] 
            
    # Calculate
    n = len(classes)
    for i in range( n ):
        w_c[mask == i] = 1 / (n*frecs[i])
    
    return w_c

    
