
import numpy as np
import numpy.matlib as mth


#-------------------------------------------------------------------
# Math
#-------------------------------------------------------------------

# R_t = angle2mat( \fi_t^i )
# Exemplo:
# R = angle2mat([0, 0, 0])
# print(np.sum(R==np.eye(3)) == 9)
#
def angle2mat( fi ):
    "To convert angle to rotation matrix"
    rx = fi[0]; ry = fi[1]; rz = fi[2];
    sx = np.sin(rx); cx = np.cos(rx);
    sy = np.sin(ry); cy = np.cos(ry);
    sz = np.sin(rz); cz = np.cos(rz); 
    Rx = np.array([[ 1,   0,  0], [ 0, cx, -sx], [  0, sx, cx]]);
    Ry = np.array([[cy,   0, sy], [ 0,  1,   0], [-sy,  0, cy]]); 
    Rz = np.array([[cz, -sz,  0], [sz, cz,   0], [  0,  0,  1]]);
    return np.dot(Rz,np.dot(Ry,Rx));


# function pp = tocenter(p)
def center(p):
    "Center point"
    c = np.mean(p,axis=0);
    return p-mth.repmat(c,p.shape[0],1);
     
# function p=projection(P,K,R,T)
# Exemplo:
# P = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]);
# K = np.array([[10, 0, 10],[0, 10, 10],[0, 0, 1]]);
# R = angle2mat([0,0,np.pi/4]);
# t = np.array([[0,0,-10]]);
# p = projection(P,K,R,t);
# plt.figure(1)
# plt.plot(p[:,0],p[:,1],'o');
# plt.show();
#
def projection(P,K,R,t):
    "Projection in 2d plane"
    P = center(P);
    H = np.dot(K,np.concatenate((R,t.T),axis=1));
    P = np.concatenate((P,np.ones([P.shape[0],2])),axis=1);
    p = np.dot(H,P.T);
    p = p[0:2]/p[2,:];     
    return p.T;


# function bbox = boundingbox(box)
# minx = min(box(:,1)); miny = min(box(:,2));
# maxx = max(box(:,1)); maxy = max(box(:,2));
# bbox = [minx miny; maxx maxy];
# end
def boundingbox(box):
    "Estimate boundingbox"
    xmin = np.min(box[:,0]); ymin = np.min(box[:,1]);
    xmax = np.max(box[:,0]); ymax = np.max(box[:,1]);
    bbox = np.array([[xmin, ymin],[xmax, ymax]]);
    return bbox;

def validimagebox(box, imshape):
    '''get valid box in the image'''
    xmin = np.max((box[0,0],0)); ymin = np.max((box[0,1],0));
    xmax = np.min((box[1,0],imshape[1])); ymax = np.min((box[1,1],imshape[0]));
    box = np.array([[xmin, ymin],[xmax, ymax]]);
    return box;

def boxdimension(bbox):
    h = abs(bbox[1,1] - bbox[0,1]); 
    w = abs(bbox[1,0] - bbox[0,0]);
    return w, h

def isboxtruncate(box, imsize):
    "Is box truncate for the image"
    return (np.any(box<0) or np.any(box[:,0]>=imsize[1]) or np.any(box[:,1]>=imsize[0])); 

def isboxocclude(bbox1, bbox2):
    "Is occlude box"
    s1 = bbox1[1,:] - bbox1[0,:];
    s2 = bbox2[1,:] - bbox2[0,:]; 
    c1 = np.mean(bbox1,axis=0);
    c2 = np.mean(bbox2,axis=0);
    d = np.abs(c1-c2);
    s = (s1+s2)/2;
    return np.all((d-s)+0.5 < 0);

def boxocclude(bbox1, bbox2):
    "Is occlude box"
    s1 = bbox1[1,:] - bbox1[0,:];
    s2 = bbox2[1,:] - bbox2[0,:]; 
    c1 = np.mean(bbox1,axis=0);
    c2 = np.mean(bbox2,axis=0);
    d = np.abs(c1-c2);
    s = (s1+s2)/2;
    return d/s;

def bboxadjust(bbox, aspX=1.0, aspY=1.0, minX=0.0, minY=0.0):
    p=bbox[4];   
    bbox = np.array([[bbox[0], bbox[1]],[bbox[2], bbox[3]]]);            
    bbox[:,0] = bbox[:,0]*(1/aspX)  + minX;
    bbox[:,1] = bbox[:,1]*(1/aspY)  + minY;
    o = np.array([bbox[0,0], bbox[0,1], bbox[1,0], bbox[1,1], p ]);  
    return list(o.tolist());






    
