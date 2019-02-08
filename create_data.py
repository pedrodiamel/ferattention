


import os
import sys
import cv2
import numpy as np
from tqdm  import tqdm

from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view

sys.path.append('../')
from torchlib.datasets.factory  import FactoryDataset 
from torchlib.datasets import Dataset


import math
import multiprocessing
import scipy.misc
from pytvision.transforms import functional as F


def getmask( x, p ):
    mask = np.zeros( x.shape[:2] )
    p = [  [[int(e[0]),int(e[1]) ]]  for e in p ]
    hull = cv2.convexHull( np.array(p),  False  )
    cv2.fillPoly(mask, [hull], 1)   
    return mask

def param2theta(mat_r, w, h):
    H = np.concatenate( (mat_r,[[0,0,1]]),axis=0 )      
    param = H #np.linalg.inv(H)
    theta = np.zeros([2,3])
    theta[0,0] = param[0,0]
    theta[0,1] = param[0,1]*h/w
    theta[0,2] = param[0,2]*2/w + param[0,0] + param[0,1] - 1
    theta[1,0] = param[1,0]*w/h
    theta[1,1] = param[1,1]
    theta[1,2] = param[1,2]*2/h + param[1,0] + param[1,1] - 1
    return theta

def angle2mat( fi ):
    "To convert angle to rotation matrix"
    rx = fi[0]; ry = fi[1]; rz = fi[2];
    sx = np.sin(rx); cx = np.cos(rx)
    sy = np.sin(ry); cy = np.cos(ry)
    sz = np.sin(rz); cz = np.cos(rz)
    Rx = np.array([[ 1,   0,  0], [ 0, cx, -sx], [  0, sx, cx]])
    Ry = np.array([[cy,   0, sy], [ 0,  1,   0], [-sy,  0, cy]]) 
    Rz = np.array([[cz, -sz,  0], [sz, cz,   0], [  0,  0,  1]])
    return np.dot(Rz,np.dot(Ry,Rx))

def boundingbox(box):
    "Estimate boundingbox"
    xmin = np.min(box[:,0]); ymin = np.min(box[:,1])
    xmax = np.max(box[:,0]); ymax = np.max(box[:,1])
    bbox = np.array([[xmin, ymin],[xmax, ymax]])
    return bbox


def resize( image, height=128,  width=128, interpolate_mode=cv2.INTER_LANCZOS4 ):
    return F.resize_image(image, 
        height=height, width=width, 
        resize_mode='square', 
        padding_mode=cv2.BORDER_CONSTANT,
        interpolate_mode=interpolate_mode, 
        )

def adjust( image, plm  ):
    
    mu = plm.mean(axis=0)
    plm_c = plm - mu

    p1 = [ *plm_c[39,:], 1 ] 
    p2 = [ *plm_c[42,:], 1 ] 
    l = np.cross(p1, p2)
    th =  math.atan2( l[1]/l[2], l[0]/l[2] )

    #print(90 - th* 180 / math.pi)
    d = ( (p1[0] - p2[0] )**2 + (p1[1] - p2[1] )**2 ) ** 0.5
    #print(d*0.2)
    
    ang = 90 - th* 180 / math.pi
    imsize = image.shape     

    matR = cv2.getRotationMatrix2D( (imsize[1]//2, imsize[0]//2) , -ang ,1 ) 
    H = param2theta(matR, imsize[1], imsize[0]  )

    image_rot = cv2.warpAffine(image, matR, (imsize[1], imsize[0]))
    plm_rot = np.dot( H, np.concatenate((plm_c,np.ones([plm_c.shape[0],1])),axis=1).T )   
    plm_rot = plm_rot.T
    plm_rot = plm_rot + mu
    
    bbox = boundingbox( plm_rot )
    
    br = d*0.2
    bbox[0,0] = bbox[0,0] - br if bbox[0,0] - br > 0 else 0    
    bbox[0,1] = bbox[0,1] - br if bbox[0,1] - br > 0 else 0
    bbox[1,0] = bbox[1,0] + br if bbox[1,0] + br < imsize[0] else imsize[1]
    bbox[1,1] = bbox[1,1] + br if bbox[1,1] + br < imsize[1] else imsize[0]
        
        
    image_rot = image_rot[  int(bbox[0,1]):int(bbox[1,1]), int(bbox[0,0]):int(bbox[1,0]), : ]
    plm_rot[:,0] = plm_rot[:,0] - bbox[0,0]
    plm_rot[:,1] = plm_rot[:,1] - bbox[0,1]
    
    return image_rot, plm_rot
 

    
def create_dataset():
    
    pathname = '~/.datasets'
    pathname = os.path.expanduser( pathname )
    dataset  = FactoryDataset.affect
    newname  = 'affectnetdarck'
    subset   = FactoryDataset.training
    height   = 128
    width    = 128
    
    pathnewdataset = os.path.join(pathname, dataset) #'./out'
    
    dataset=FactoryDataset.factory(
        pathname=pathname, 
        name=dataset, 
        subset=subset, 
        download=True 
        )
    
        
    print('Length dataset: ', len(dataset))
    
    indexs_exclude = []
    for i in tqdm(range( len(dataset) )):
    
        try:
            image, label = dataset[ i ]
            plm = np.array(dataset.getlandamarks()).reshape( -1, 2 )

            image_rot, plm_rot = adjust( image, plm )
            mask_rot   = getmask( image_rot, plm_rot )
            image_rot  = resize( image_rot, height=height,  width=width, interpolate_mode=cv2.INTER_LANCZOS4 )
            mask_rot   = resize( mask_rot , height=height,  width=width, interpolate_mode=cv2.INTER_NEAREST )[:,:,0].astype(np.uint8)
            image_rot  = cv2.cvtColor(image_rot, cv2.COLOR_RGB2GRAY)
            image_mask = ( image_rot * mask_rot )
            
            #print(image_mask.min(), image_mask.max())            
            
            if mask_rot.sum() < 10:
                print('{};{}'.format(i,dataset.index) )
                indexs_exclude.append( (i,dataset.index) )
                continue
                        
            # save        
            path, name = dataset.getpathname().split('/')
            newpath = os.path.join( pathnewdataset, newname, path )
            if not os.path.exists( newpath ):
                os.makedirs(newpath)
                
            cv2.imwrite( os.path.join(newpath,  '{}.png'.format( name.split('.')[0])  ), image_mask )
            #scipy.misc.imsave( os.path.join(newpath, name), image_mask  ) #name , '{}.png'.format( name.split('.')[0]))
            
                        
        except:
            print('{};{}'.format(i,dataset.index) )
            indexs_exclude.append( (i,dataset.index) )
        
        
#         if i==100:
#             break
    
    
    #indexs_exclude = np.concatenate(indexs_exclude, axis=0)
    print(indexs_exclude)
    np.savetxt('./out/cratedataset.txt', indexs_exclude, delimiter=';',  fmt='%d') 
    

if __name__ == '__main__':
    create_dataset()