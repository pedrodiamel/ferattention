import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def attLoss( x_org, y_mask, att ):    
    loss_att = ((( (x_org*y_mask[:,1,...].unsqueeze(dim=1)) - att ) ** 2)).mean()
    #loss_att =  ((( (x_org*y_mask[:,1,...].unsqueeze(dim=1)).mean(dim=1) - att.mean(dim=1) ) ** 2)).mean()
    #loss_att = (((x_org*y_mask[:,1,...].unsqueeze(dim=1) - att ) ** 2) * ( y_mask[:,0,...].unsqueeze(dim=1) + y_mask[:,1,...].unsqueeze(dim=1)*0.5  )).mean()  
    #loss_att = ( torch.abs(att*y_mask[:,0,...].unsqueeze(dim=1)) ).sum() / y_mask[:,0,...].sum()      
    loss_att = torch.clamp(loss_att, max=30)
    return loss_att


class WeightedMCEloss(nn.Module):

    def __init__(self ):
        super(WeightedMCEloss, self).__init__()

    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h )
        weight = centercrop(weight, w, h )
        
        y_pred_log =  F.log_softmax(y_pred, dim=1)
        logpy = torch.sum( weight * y_pred_log * y_true, dim=1 )
        #loss  = -torch.sum(logpy) / torch.sum(weight)
        loss  = -torch.mean(logpy)
        return loss


class WeightedMCEFocalloss(nn.Module):
    
    def __init__(self, gamma=2.0 ):
        super(WeightedMCEFocalloss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h )
        weight = centercrop(weight, w, h )
        
        y_pred_log =  F.log_softmax(y_pred, dim=1)

        fweight = (1 - F.softmax(y_pred, dim=1) ) ** self.gamma
        weight  = weight*fweight

        logpy = torch.sum( weight * y_pred_log * y_true, dim=1 )
        #loss  = -torch.sum(logpy) / torch.sum(weight)
        loss  = -torch.mean(logpy)
        
        return loss

class WeightedBCELoss(nn.Module):
    
    def __init__(self ):
        super(WeightedBCELoss, self).__init__()

    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        weight = centercrop(weight, w, h )
     
        logit_y_pred = torch.log(y_pred / (1. - y_pred))
        loss = weight * (logit_y_pred * (1. - y_true) + 
                        torch.log(1. + torch.exp(-torch.abs(logit_y_pred))) + torch.clamp(-logit_y_pred, min=0.))
        loss = torch.sum(loss) / torch.sum(weight)

        return loss

class BCELoss(nn.Module):
    
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true ):        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        loss = self.bce(y_pred, y_true)
        return loss

class WeightedBDiceLoss(nn.Module):
    
    def __init__(self ):
        super(WeightedBDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        weight = centercrop(weight, w, h )
        y_pred = self.sigmoid(y_pred)
        smooth = 1.
        w, m1, m2 = weight, y_true, y_pred
        score = (2. * torch.sum(w * m1 * m2) + smooth) / (torch.sum(w * m1) + torch.sum(w * m2) + smooth)
        loss = 1. - torch.sum(score)
        return loss


class BDiceLoss(nn.Module):
    
    def __init__(self):
        super(BDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true, weight=None ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        y_pred = self.sigmoid(y_pred)

        smooth = 1.
        y_true_f = flatten(y_true)
        y_pred_f = flatten(y_pred)
        score = (2. * torch.sum(y_true_f * y_pred_f) + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return 1. - score


    
class BLogDiceLoss(nn.Module):
    
    def __init__(self, classe = 1 ):
        super(BLogDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classe = classe

    def forward(self, y_pred, y_true, weight=None ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        y_pred = self.sigmoid(y_pred)

        eps = 1e-15
        dice_target = (y_true[:,self.classe,...] == 1).float()
        dice_output = y_pred[:,self.classe,...]
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + eps

        return -torch.log(2 * intersection / union)

class WeightedMCEDiceLoss(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=1.0  ):
        super(WeightedMCEDiceLoss, self).__init__()
        self.loss_mce = WeightedMCEFocalloss()
        self.loss_dice = BLogDiceLoss()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight ):        
        
        alpha = self.alpha
        weight = torch.pow(weight,self.gamma)
        loss_dice = self.loss_dice(y_pred, y_true)
        loss_mce = self.loss_mce(y_pred, y_true, weight)
        loss = loss_mce + alpha*loss_dice        
        return loss

class MCEDiceLoss(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=1.0  ):
        super(MCEDiceLoss, self).__init__()
        self.loss_mce = BCELoss()
        self.loss_dice = BLogDiceLoss( classe=1  )
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight=None ):        
        
        alpha = self.alpha  

        # bce(all_channels) +  dice_loss(mask_channel) + dice_loss(border_channel)   
        loss_all  = self.loss_mce( y_pred[:,:2,...], y_true[:,:2,...])    
        loss_fg   = self.loss_dice( y_pred, y_true )
        
        #loss_fg   = self.loss_dice( y_pred[:,1,...].unsqueeze(1), y_true[:,1,...].unsqueeze(1) )
        #loss_th   = self.loss_dice( y_pred[:,2,...].unsqueeze(1), y_true[:,2,...].unsqueeze(1) )
        #loss = loss_all + loss_fg + loss_th           
        loss = loss_all + 2.0*loss_fg  

        return loss


class Accuracy(nn.Module):
    
    def __init__(self, bback_ignore=True):
        super(Accuracy, self).__init__()
        self.bback_ignore = bback_ignore 

    def forward(self, y_pred, y_true ):
        
        n, ch, h, w = y_pred.size()        
        y_true = centercrop(y_true, w, h)

        prob = F.softmax(y_pred, dim=1).data
        prediction = torch.argmax(prob,1)

        accs = []
        for c in range( int(self.bback_ignore), ch ):
            yt_c = y_true[:,c,...]
            num = (((prediction.eq(c) + yt_c.data.eq(1)).eq(2)).float().sum() + 1 )
            den = (yt_c.data.eq(1).float().sum() + 1)
            acc = (num/den)*100
            accs.append(acc)
        
        accs = torch.stack(accs)
        return accs.mean()


class Dice(nn.Module):
    
    def __init__(self, bback_ignore=True):
        super(Dice, self).__init__()
        self.bback_ignore = bback_ignore       

    def forward(self, y_pred, y_true ):
        
        eps = 1e-15
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)

        prob = F.softmax(y_pred, dim=1)
        prob = prob.data
        prediction = torch.argmax(prob, dim=1)

        y_pred_f = flatten(prediction).float()
        dices = []
        for c in range(int(self.bback_ignore), ch ):
            y_true_f = flatten(y_true[:,c,...]).float()
            intersection = y_true_f * y_pred_f
            dice = (2. * torch.sum(intersection) / ( torch.sum(y_true_f) + torch.sum(y_pred_f) + eps ))*100
            dices.append(dice)
        
        dices = torch.stack(dices)
        return dices.mean()


def centercrop(image, w, h):        
    nt, ct, ht, wt = image.size()
    padw, padh = (wt-w) // 2 ,(ht-h) // 2
    if padw>0 and padh>0: image = image[:,:, padh:-padh, padw:-padw]
    return image

def flatten(x):        
    x_flat = x.clone()
    x_flat = x_flat.view(x.shape[0], -1)
    return x_flat


def to_gpu( x, cuda ):
    return x.cuda() if cuda else x

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels.long()]     # [N,D]



# GMM Loss metrics 

class DGMMLoss(nn.Module):
    r"""DGMMLoss class
    Deep Gaussian Mixture Model in embedded space loss 

    L = || P(w_j | f(x), mu_j, Sigma_j ) - P(w_j, x) ||_2 

    Args:
        classes (int): number of classes
        sigma (float):
        cuda (bool):
        mix (bool): mixup
    """
    
    def __init__(self, classes=10, sigma=1.0, cuda=False, mix=False, knn=False  ):
        super(DGMMLoss, self).__init__()
        self.classes = classes
        self.sigma   = sigma 
        self.cuda    = cuda
        self.mix     = mix
        self.knn     = knn         
        
        
    def forward( self, x, y, classes=None ):                
        
        if self.mix:
            # regeneration 
            alpha=2.0
            k=11
            
            lam = torch.distributions.beta.Beta(torch.Tensor([alpha]), torch.Tensor([alpha])).sample()
            indices = torch.randperm(x.shape[0])  
            if self.cuda: lam = lam.cuda()  
            x_ul = x * lam.expand_as(x) + x[indices,...] * (1 - lam.expand_as(x))
            
            y_ul = []
            for xi in x_ul:
                d = torch.sum((xi - x)**2 ,dim=1).squeeze() 
                #i = torch.argmin(d, dim=0) #1-NN
                i = torch.topk(d,k=k,dim=0,largest=False)[1] #k-NN
                yi = torch.mode( y[i], dim=0)[0]
                y_ul.append( yi )
        
            y_ul = torch.stack(y_ul,dim=0)         
            y_ul = to_gpu(y_ul, self.cuda) 
            
            x = torch.cat( [x, x_ul], dim=0 )
            y = torch.cat( [y, y_ul], dim=0 )
          
        
        classes = self.classes if not classes else classes
        num = x.shape[0]
        dim = x.shape[1]       
        
        
        ## initialization        
        pi = to_gpu(torch.zeros( (num, classes ) ), self.cuda ) 
        y  = to_gpu(y, self.cuda )  
        yh = to_gpu(torch.zeros( (num, classes ) ), self.cuda ).float()   
        
        for i, c in enumerate( range(classes) ): 
            index = y == int(c)
            if index.sum() == 0: 
                continue
            xc = x[ index, ...]            
            muc = xc.mean(dim=0)
            pi[:,i] = torch.exp(-torch.sum((x - muc)**2 ,dim=1).squeeze() / ( 2.0*self.sigma**2) )            
            yh[index, i] = 1
            
        eps = 1e-15
        s = torch.sum(pi, dim=1).unsqueeze(1)    
        pi = pi/( s.expand_as(pi) + eps )
        pi = pi.clamp(0.0,1.0)
        
        ## loss function        
        loss_gm = torch.mean( torch.sum( (pi-yh.float())**2, dim=1 ).squeeze() )   
                
        # loss 
        loss = loss_gm
        
        if self.knn:
            ## Knn loss
            k=3
            y_ng = []
            for ix in x:
                d  = torch.sum((x - ix)**2 ,dim=1).squeeze()             
                #i = torch.argmin(d, dim=0) #1-NN
                i  = torch.topk(d, k=k+1, dim=0, largest=False )[1] #1-NN
                yi = torch.mode( y[i[1:]], dim=0)[0]
                #yi = y[i[1]]
                y_ng.append( yi )        

            y_ng = torch.stack(y_ng, dim=0)         
            y_ng = to_gpu(y_ng, self.cuda) 
            loss_knn = torch.nn.functional.mse_loss( y_ng, y )
            # loss 
            loss = loss + 0.01*loss_knn
        
        return loss


class GMMAccuracy(nn.Module):
    
    def __init__(self, classes=10, cuda=False):
        super(GMMAccuracy, self).__init__()
        self.classes = classes
        self.cuda = cuda
        
    def forward( self, x, y, classes=None ):  

        #numclasses = self.classes
        #numclasses = self.classes if not classes else classes        
        classes  = np.unique(y)
        numclasses = len(classes)
        
        num = x.shape[0]
        dim = x.shape[1]
        
        ## initialization
        Xmu = to_gpu(torch.zeros((num, numclasses )), self.cuda)  
        y   = to_gpu(y, self.cuda ).int()
        yh  = to_gpu(torch.zeros( (num, ) ), self.cuda ).int()
        
        ## Ecuation
        for i, c in enumerate( classes ):
            index = y==int(c)      
            if index.sum() == 0: continue            
            xc = x[ index , ... ]
            muc = xc.mean(dim=0)        
            Xmu[:,i] = torch.sum( (x - muc)**2 , dim=1)
            yh[index] = i
        
        pred = torch.argmin(Xmu, 1).int()
        correct = pred.eq( yh ).int()      
        acc = (correct.sum(0, keepdim=True).float().mul_(100.0 / num))    
        
        return acc
        


## Baseline clasification

class TopkAccuracy(nn.Module):
    
    def __init__(self, topk=(1,)):
        super(TopkAccuracy, self).__init__()
        self.topk = topk

    def forward(self, output, target):
        """Computes the precision@k for the specified values of k"""
        
        maxk = max(self.topk)
        batch_size = target.size(0)

        pred = output.topk(maxk, 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append( correct_k.mul_(100.0 / batch_size) )

        return res



class ConfusionMeter( object ):
    """Maintains a confusion matrix for a given calssification problem.
    https://github.com/pytorch/tnt/tree/master/torchnet/meter

    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.

    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not

    """

    def __init__(self, k, normalized=False):
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes

        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors

        """

        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf



