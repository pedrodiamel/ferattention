
import cv2
from torchvision import transforms
from pytvision.transforms import transforms as mtrans


# transformations 
#normalize = mtrans.ToMeanNormalization(
#    #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
#    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
#    )

# cifar10
normalize = mtrans.ToMeanNormalization(
    mean = (0.4914, 0.4822, 0.4465), #[x / 255 for x in [125.3, 123.0, 113.9]],
    std  = (0.2023, 0.1994, 0.2010), #[x / 255 for x in [63.0, 62.1, 66.7]],
    )

# cifar100
#normalize = mtrans.ToMeanNormalization(
#    mean = [x / 255 for x in [129.3, 124.1, 112.4]],
#    std = [x / 255 for x in [68.2, 65.4, 70.4]],
#    )

# svhn
#normalize = mtrans.ToMeanNormalization(
#    mean = [x / 255 for x in [127.5, 127.5, 127.5]],
#    std = [x / 255 for x in [127.5, 127.5, 127.5]],
#    )


# normalize = mtrans.ToNormalization()

def get_transforms_aug( size_input ):
    return transforms.Compose([        
        
        #------------------------------------------------------------------
        #Resize 
        mtrans.ToResize( (size_input,size_input), resize_mode='square', padding_mode=cv2.BORDER_REPLICATE),  
        
        #------------------------------------------------------------------
        #Colors           
        mtrans.ToRandomTransform( mtrans.RandomBrightness( factor=0.25 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomContrast( factor=0.25 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomGamma( factor=0.25 ), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.RandomRGBPermutation(), prob=0.50 ),
        mtrans.ToRandomTransform( mtrans.CLAHE(), prob=0.25 ),
        mtrans.ToRandomTransform(mtrans.ToGaussianBlur( sigma=0.07 ), prob=0.25 ),
        #mtrans.ToRandomTransform(mtrans.ToGaussianNoise( sigma=0.05 ), prob=0.25 ),
       
        #------------------------------------------------------------------
        mtrans.ToTensor(),
        normalize,
        
        ])    
    


def get_transforms_det(size_input):    
    return transforms.Compose([
        mtrans.ToResize( (size_input, size_input), resize_mode='squash' ) ,
        #mtrans.ToResize( (size_input, size_input), resize_mode='square', padding_mode=cv2.BORDER_REPLICATE ) ,
        mtrans.ToTensor(),
        normalize,
        ])
     

