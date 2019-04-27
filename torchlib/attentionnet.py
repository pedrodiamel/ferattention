

import os
import math
import shutil
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
from tqdm import tqdm

from . import models as nnmodels
from . import netlosses as nloss

from pytvision.neuralnet import NeuralNetAbstract
from pytvision.logger import Logger, AverageFilterMeter, AverageMeter
from pytvision import graphic as gph
from pytvision import netlearningrate
from pytvision import utils as pytutils

#----------------------------------------------------------------------------------------------
# Neural Net for Attention


class AttentionNeuralNetAbstract(NeuralNetAbstract):
    """
    Attention Neural Net Abstract
    """

    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0,
        view_freq=1
        ):
        """
        Initialization
            -patchproject (str): path project
            -nameproject (str):  name project
            -no_cuda (bool): system cuda (default is True)
            -parallel (bool)
            -seed (int)
            -print_freq (int)
            -gpu (int)
            -view_freq (in epochs)
        """

        super(AttentionNeuralNetAbstract, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu  )
        self.view_freq = view_freq

 
    def create(self, 
        arch, 
        num_output_channels,
        num_input_channels,        
        loss,
        lr,
        optimizer,
        lrsch,
        momentum=0.9,
        weight_decay=5e-4,
        pretrained=False,
        size_input=388,
        num_classes=8,
        ):
        """
        Create            
            -arch (string): architecture
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) : 
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
        """
        
        cfg_opt={ 'momentum':momentum, 'weight_decay':weight_decay } 
        cfg_scheduler={ 'step_size':100, 'gamma':0.1  }        
        self.num_classes = num_classes
        
        super(AttentionNeuralNetAbstract, self).create( 
            arch, 
            num_output_channels, 
            num_input_channels, 
            loss, 
            lr,  
            optimizer, 
            lrsch, 
            pretrained,
            cfg_opt=cfg_opt, 
            cfg_scheduler=cfg_scheduler,            
        )
        
        self.size_input = size_input
        
        self.accuracy = nloss.Accuracy()
        self.topk     = nloss.TopkAccuracy()
        self.gmm      = nloss.GMMAccuracy( classes=num_classes, cuda=self.cuda  )
        self.dice     = nloss.Dice()
       
        # Set the graphic visualization
        self.visheatmap = gph.HeatMapVisdom(env_name=self.nameproject, heatsize=(100,100) )


        

    # def test(self, data_loader ):        
    #     # initialization
    #     k, masks, ids = 0, [], []
    #     # switch to evaluate mode
    #     self.net.eval()
    #     with torch.no_grad():
    #         end = time.time()
    #         for i, (idd, x_img) in enumerate( tqdm(data_loader) ):                
    #             x = x_img.cuda() if self.cuda else x_img                    
    #             # fit (forward)
    #             y_hat = self.net(x)
    #             y_hat = F.softmax(y_hat, dim=1)    
    #             y_hat = pytutils.to_np(y_hat)
    #             masks.append( y_hat )
    #             ids.append( idd )      
    #     return ids, masks

    def representation( self, dataloader, breal=True ):
        Y_labs = []
        Y_lab_hats = []
        Zs = []         
        self.net.eval()
        with torch.no_grad():
            for i_batch, sample in enumerate( tqdm(dataloader) ):

                if breal:                 
                    x_img, y_lab = sample['image'], sample['label'].argmax(dim=1)
                else:
                    x_org, x_img, y_mask, y_lab = sample
                    y_lab=y_lab[:,0]

                x_img = x_img.cuda() if self.cuda else x_img
                z, y_lab_hat, _,_,_,_,_ = self.net( x_img )
                Y_labs.append(y_lab)
                Y_lab_hats.append(y_lab_hat.data.cpu())
                Zs.append(z.data.cpu())
                
        Y_labs = np.concatenate( Y_labs, axis=0 )
        Y_lab_hats = np.concatenate( Y_lab_hats, axis=0 )
        Zs = np.concatenate( Zs, axis=0 )        
        return Y_labs, Y_lab_hats, Zs
    
    def __call__(self, image ):        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image    
            z, y_lab_hat, att, theta, att_t, fmap, srf = self.net(x)                         
            y_lab_hat = F.softmax( y_lab_hat, dim=1 )            
        return z, y_lab_hat, att, theta, att_t, fmap, srf

    def _create_model(self, arch, num_output_channels, num_input_channels, pretrained ):
        """
        Create model
            -arch (string): select architecture
            -num_classes (int)
            -num_channels (int)
            -pretrained (bool)
        """    

        self.net = None    

        #-------------------------------------------------------------------------------------------- 
        # select architecture
        #--------------------------------------------------------------------------------------------
        kw = {'dim': num_output_channels, 'num_classes': self.num_classes, 'num_channels': num_input_channels, 'pretrained': pretrained}
        self.net = nnmodels.__dict__[arch](**kw)
        
        self.s_arch = arch
        self.num_output_channels = num_output_channels
        self.num_input_channels = num_input_channels

        if self.cuda == True:
            self.net.cuda()
        if self.parallel == True and self.cuda == True:
            self.net = nn.DataParallel(self.net, device_ids= range( torch.cuda.device_count() ))
              

    def save(self, epoch, prec, is_best=False, filename='checkpoint.pth.tar'):
        """
        Save model
        """
        print('>> save model epoch {} ({}) in {}'.format(epoch, prec, filename))
        net = self.net.module if self.parallel else self.net
        pytutils.save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': self.s_arch,
                'imsize': self.size_input,
                'num_output_channels': self.num_output_channels,
                'num_input_channels': self.num_input_channels,
                'num_classes': self.num_classes,
                'state_dict': net.state_dict(),
                'prec': prec,
                'optimizer' : self.optimizer.state_dict(),
            }, 
            is_best,
            self.pathmodels,
            filename
            )

    def load(self, pathnamemodel):
        bload = False
        if pathnamemodel:
            if os.path.isfile(pathnamemodel):
                print("=> loading checkpoint '{}'".format(pathnamemodel))
                checkpoint = torch.load( pathnamemodel ) if self.cuda else torch.load( pathnamemodel, map_location=lambda storage, loc: storage )                
                self.num_classes = checkpoint['num_classes']
                self._create_model(checkpoint['arch'], checkpoint['num_output_channels'], checkpoint['num_input_channels'], False )                
                self.size_input = checkpoint['imsize'] 
                self.net.load_state_dict( checkpoint['state_dict'] )              
                print("=> loaded checkpoint for {} arch!".format(checkpoint['arch']))
                bload = True
            else:
                print("=> no checkpoint found at '{}'".format(pathnamemodel))        
        return bload            

class AttentionNeuralNet(AttentionNeuralNetAbstract):
    """
    Attention Neural Net 
    Args:
        -patchproject (str): path project
        -nameproject (str):  name project
        -no_cuda (bool): system cuda (default is True)
        -parallel (bool)
        -seed (int)
        -print_freq (int)
        -gpu (int)
        -view_freq (in epochs)
    """
    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0,
        view_freq=1
        ):
        super(AttentionNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu, view_freq  )
        
 
    def create(self, 
        arch, 
        num_output_channels,
        num_input_channels,        
        loss,
        lr,
        optimizer,
        lrsch,
        momentum=0.9,
        weight_decay=5e-4,
        pretrained=False,
        size_input=388,
        num_classes=8,
        ):
        """
        Create    
        Args:        
            -arch (string): architecture
            -num_output_channels,
            -num_input_channels,  
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) : 
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
            -
        """        
        super(AttentionNeuralNet, self).create( 
            arch, 
            num_output_channels,
            num_input_channels,        
            loss,
            lr,
            optimizer,
            lrsch,
            momentum,
            weight_decay,
            pretrained,
            size_input,
            num_classes,          
        )
        
        self.logger_train = Logger( 'Train', ['loss', 'loss_bce', 'loss_att' ], [ 'topk'], self.plotter  )
        self.logger_val   = Logger( 'Val  ', ['loss', 'loss_bce', 'loss_att' ], [ 'topk'], self.plotter )
      
    
    def training(self, data_loader, epoch=0):        

        #reset logger
        self.logger_train.reset()
        data_time  = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, (x_org, x_img, y_mask, meta ) in enumerate(data_loader):
            
            # measure data loading time
            data_time.update(time.time() - end)
            batch_size = x_img.shape[0]
            
            y_lab = meta[:,0]
            y_theta   = meta[:,1:].view(-1, 2, 3)            

            if self.cuda:
                x_org   = x_org.cuda()
                x_img   = x_img.cuda() 
                y_mask  = y_mask.cuda() 
                y_lab   = y_lab.cuda()
                y_theta = y_theta.cuda()
            
            # fit (forward)            
            y_lab_hat, att, _, _ = self.net( x_img, x_org )                
            
            # measure accuracy and record loss           
            loss_bce  = self.criterion_bce( y_lab_hat, y_lab.long() )           
            loss_att  = self.criterion_att( x_org, y_mask, att )    
            loss      = loss_bce + loss_att           
            topk      = self.topk( y_lab_hat, y_lab.long() )
            
            # optimizer
            self.optimizer.zero_grad()
            (loss*batch_size).backward()
            self.optimizer.step()
            
            # update
            self.logger_train.update(
                {'loss': loss.cpu().item(), 'loss_bce': loss_bce.cpu().item(), 'loss_att':loss_att.cpu().item() },
                {'topk': topk[0][0].cpu() },
                batch_size,
                )
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.print_freq == 0:  
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )

    def evaluate(self, data_loader, epoch=0):
        
        # reset loader
        self.logger_val.reset()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, (x_org, x_img, y_mask, meta) in enumerate(data_loader):
                
                # get data (image, label)
                batch_size = x_img.shape[0]  
                                               
                y_lab = meta[:,0]
                y_theta   = meta[:,1:].view(-1, 2, 3)
                                
                if self.cuda:
                    x_org   = x_org.cuda()
                    x_img   = x_img.cuda()
                    y_mask  = y_mask.cuda()
                    y_lab   = y_lab.cuda()
                    y_theta = y_theta.cuda()
                              

                # fit (forward)            
                y_lab_hat, att, fmap, srf  = self.net( x_img, x_org ) 
                                
                # measure accuracy and record loss       
                loss_bce  = self.criterion_bce(  y_lab_hat, y_lab.long() )
                loss_att  = self.criterion_att( x_org, y_mask, att )
                loss      = loss_bce + loss_att          
                topk      = self.topk( y_lab_hat, y_lab.long() )               
                 
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                                
                # update
                self.logger_val.update( 
                    {'loss': loss.cpu().item(), 'loss_bce': loss_bce.cpu().item(), 'loss_att':loss_att.cpu().item() },
                    {'topk': topk[0][0].cpu() },    
                    batch_size,          
                    )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader), 
                        batch_time, 
                        bplotter=False,
                        bavg=True, 
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['topk'].avg

        self.logger_val.logger(
            epoch, epoch, i, len(data_loader), 
            batch_time,
            bplotter=True,
            bavg=True, 
            bsummary=True,
            )

        #vizual_freq
        if epoch % self.view_freq == 0:

            att   = att[0,:,:,:].permute( 1,2,0 ).mean(dim=2)
            srf   = srf[0,:,:,:].permute( 1,2,0 ).sum(dim=2)  
            fmap  = fmap[0,:,:,:].permute( 1,2,0 ).mean(dim=2) 
                                    
            self.visheatmap.show('Image', x_img.data.cpu()[0].numpy()[0,:,:])           
            self.visheatmap.show('Image Attention',att.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Feature Map',srf.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Attention Map',fmap.cpu().numpy().astype(np.float32) )

        return acc

    def representation( self, dataloader, breal=True ):
        Y_labs = []
        Y_lab_hats = []       
        self.net.eval()
        with torch.no_grad():
            for i_batch, sample in enumerate( tqdm(dataloader) ):

                if breal:                 
                    x_img, y_lab = sample['image'], sample['label']
                    y_lab = y_lab.argmax(dim=1)
                else:
                    x_org, x_img, y_mask, y_lab = sample
                    y_lab=y_lab[:,0]

                if self.cuda:
                    x_img = x_img.cuda()
                y_lab_hat, _,_,_ = self.net( x_img )
                Y_labs.append(y_lab)
                Y_lab_hats.append(y_lab_hat.data.cpu())
                                
        Y_labs = np.concatenate( Y_labs, axis=0 )
        Y_lab_hats = np.concatenate( Y_lab_hats, axis=0 )     
        return Y_labs, Y_lab_hats 
    
    
    def __call__(self, image ):        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image    
            y_lab_hat, att, fmap, srf = self.net(x)                         
            y_lab_hat = F.softmax( y_lab_hat, dim=1 )            
        return y_lab_hat, att, fmap, srf


    def _create_loss(self, loss):

        # create loss
        if loss == 'attloss':            
            self.criterion_bce = nn.CrossEntropyLoss().cuda()
            self.criterion_att = nloss.Attloss()
        else:
            assert(False)

        self.s_loss = loss

class AttentionSTNNeuralNet(AttentionNeuralNet):
    """
    Attention Neural Net with STN
    Args:
        -patchproject (str): path project
        -nameproject (str):  name project
        -no_cuda (bool): system cuda (default is True)
        -parallel (bool)
        -seed (int)
        -print_freq (int)
        -gpu (int)
        -view_freq (in epochs)
    """
    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0,
        view_freq=1
        ):
        super(AttentionSTNNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu, view_freq  )
        
    def create(self, 
        arch, 
        num_output_channels,
        num_input_channels,        
        loss,
        lr,
        optimizer,
        lrsch,
        momentum=0.9,
        weight_decay=5e-4,
        pretrained=False,
        size_input=388,
        num_classes=8,
        ):
        """
        Create    
        Args:        
            -arch (string): architecture
            -num_output_channels,
            -num_input_channels,  
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) : 
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
            -
        """        
        super(AttentionSTNNeuralNet, self).create( 
            arch, 
            num_output_channels,
            num_input_channels,        
            loss,
            lr,
            optimizer,
            lrsch,
            momentum,
            weight_decay,
            pretrained,
            size_input,
            num_classes,          
        )

        self.logger_train = Logger( 'Train', ['loss', 'loss_bce', 'loss_att', 'loss_stn' ], ['topk'], self.plotter  )
        self.logger_val   = Logger( 'Val  ', ['loss', 'loss_bce', 'loss_att', 'loss_stn' ], ['topk'], self.plotter )
    
    
    def training(self, data_loader, epoch=0):        

        #reset logger
        self.logger_train.reset()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, (x_org, x_img, y_mask, meta ) in enumerate(data_loader):
            
            # measure data loading time
            data_time.update(time.time() - end)
            batch_size = x_img.shape[0]
            
            y_lab = meta[:,0]
            y_theta   = meta[:,1:].view(-1, 2, 3)            

            if self.cuda:
                x_org   = x_org.cuda()
                x_img   = x_img.cuda() 
                y_mask  = y_mask.cuda() 
                y_lab   = y_lab.cuda()
                y_theta = y_theta.cuda()
            
            # fit (forward)            
            y_lab_hat, att, theta, _, _, _ = self.net( x_img, x_img*y_mask[:,1,...].unsqueeze(dim=1) )                
            
            # measure accuracy and record loss           
            loss_bce  = self.criterion_bce( y_lab_hat, y_lab.long() )            
            loss_att  = self.criterion_att( x_org, y_mask, att )
            loss_stn  = self.criterion_stn( x_org, y_theta, theta )
            
            loss = loss_bce + loss_att + loss_stn        
            topk  = self.topk( y_lab_hat, y_lab.long() )
                     
            
            # optimizer
            self.optimizer.zero_grad()
            (loss*batch_size).backward()
            self.optimizer.step()
            
            # update
            self.logger_train.update(
                {'loss': loss.cpu().item(), 'loss_bce': loss_bce.cpu().item(), 'loss_att':loss_att.cpu().item(), 'loss_stn':loss_stn.cpu().item() },
                {'topk': topk[0][0].cpu() },
                batch_size,
                )
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.print_freq == 0:  
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )


    def evaluate(self, data_loader, epoch=0):
        
        # reset loader
        self.logger_val.reset()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, (x_org, x_img, y_mask, meta) in enumerate(data_loader):
                
                # get data (image, label)
                batch_size = x_img.shape[0]    
                                               
                y_lab = meta[:,0]
                y_theta   = meta[:,1:].view(-1, 2, 3)
                                
                if self.cuda:
                    x_org   = x_org.cuda()
                    x_img   = x_img.cuda()
                    y_mask  = y_mask.cuda()
                    y_lab   = y_lab.cuda()
                    y_theta = y_theta.cuda()
                              

                # fit (forward)            
                y_lab_hat, att, theta, att_t, fmap, srf  = self.net( x_img ) 
                
                
                # measure accuracy and record loss       
                loss_bce  = self.criterion_bce(  y_lab_hat, y_lab.long() )
                loss_att  = self.criterion_att( x_org, y_mask, att )
                loss_stn  = self.criterion_stn( x_org, y_theta, theta )
                loss      = loss_bce + loss_att + loss_stn          
                topk      = self.topk( y_lab_hat, y_lab.long() )               
                                                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                # update
                self.logger_val.update( 
                    {'loss': loss.cpu().item(), 'loss_bce': loss_bce.cpu().item(), 'loss_att':loss_att.cpu().item(), 'loss_stn':loss_stn.cpu().item() },
                    {'topk': topk[0][0].cpu() },    
                    batch_size,          
                    )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader), 
                        batch_time, 
                        bplotter=False,
                        bavg=True, 
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['topk'].avg

        self.logger_val.logger(
            epoch, epoch, i, len(data_loader), 
            batch_time,
            bplotter=True,
            bavg=True, 
            bsummary=True,
            )

        #vizual_freq
        if epoch % self.view_freq == 0:

            att   = att[0,:,:,:].permute( 1,2,0 ).mean(dim=2)
            att_t = att_t[0,:,:,:].permute( 1,2,0 ).mean(dim=2)
            srf   = srf[0,:,:,:].permute( 1,2,0 ).sum(dim=2)  
            fmap  = fmap[0,:,:,:].permute( 1,2,0 ) 
            
            print('theta')
            print(y_theta[0,:,:] )
            print(theta[0,:,:] )
                        
            self.visheatmap.show('Image', x_img.data.cpu()[0].numpy()[0,:,:])           
            self.visheatmap.show('Image Attention',att.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Image Attention Trans',att_t.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Feature Map',srf.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Attention Map',fmap.cpu().numpy().astype(np.float32) )
            
        return acc

    
    def representation( self, dataloader, breal=True ):
        Y_labs = []
        Y_lab_hats = []        
        self.net.eval()
        with torch.no_grad():
            for i_batch, sample in enumerate( tqdm(dataloader) ):

                if breal:                 
                    x_img, y_lab = sample['image'], sample['label']
                    y_lab = y_lab.argmax(dim=1)
                else:
                    x_org, x_img, y_mask, y_lab = sample
                    y_lab=y_lab[:,0]

                if self.cuda:
                    x_img = x_img.cuda()

                y_lab_hat, _,_,_,_,_ = self.net( x_img )
                Y_labs.append(y_lab)
                Y_lab_hats.append(y_lab_hat.data.cpu())
                                
        Y_labs = np.concatenate( Y_labs, axis=0 )
        Y_lab_hats = np.concatenate( Y_lab_hats, axis=0 )      
        return Y_labs, Y_lab_hats
    
    
    def __call__(self, image ):        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image    
            y_lab_hat, att, theta, att_t, fmap, srf = self.net(x)                         
            y_lab_hat = F.softmax( y_lab_hat, dim=1 )            
        return y_lab_hat, att, theta, att_t, fmap, srf


    def _create_loss(self, loss):

        # create loss
        if loss == 'attloss':            
            self.criterion_bce = nn.CrossEntropyLoss().cuda()
            self.criterion_att = nloss.Attloss()
            self.criterion_stn = nloss.STNloss()
        else:
            assert(False)

        self.s_loss = loss

class AttentionGMMNeuralNet(AttentionNeuralNetAbstract):
    """
    Attention Neural Net and GMM representation 
    Args:
        -patchproject (str): path project
        -nameproject (str):  name project
        -no_cuda (bool): system cuda (default is True)
        -parallel (bool)
        -seed (int)
        -print_freq (int)
        -gpu (int)
        -view_freq (in epochs)
    """
    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0,
        view_freq=1
        ):
        super(AttentionGMMNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu, view_freq  )
        

 
    def create(self, 
        arch, 
        num_output_channels,
        num_input_channels,        
        loss,
        lr,
        optimizer,
        lrsch,
        momentum=0.9,
        weight_decay=5e-4,
        pretrained=False,
        size_input=388,
        num_classes=8,
        ):
        """
        Create    
        Args:        
            -arch (string): architecture
            -num_output_channels,
            -num_input_channels,  
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) : 
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
            -
        """        
        super(AttentionGMMNeuralNet, self).create( 
            arch, 
            num_output_channels,
            num_input_channels,        
            loss,
            lr,
            optimizer,
            lrsch,
            momentum,
            weight_decay,
            pretrained,
            size_input,
            num_classes,          
        )

        self.logger_train = Logger( 'Train', ['loss', 'loss_gmm', 'loss_bce', 'loss_att' ], [ 'topk', 'gmm'], self.plotter  )
        self.logger_val   = Logger( 'Val  ', ['loss', 'loss_gmm', 'loss_bce', 'loss_att' ], [ 'topk', 'gmm'], self.plotter )
        
      
    def training(self, data_loader, epoch=0):        

        #reset logger
        self.logger_train.reset()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, (x_org, x_img, y_mask, meta ) in enumerate(data_loader):
            
            # measure data loading time
            data_time.update(time.time() - end)
            batch_size = x_img.shape[0]
            
            y_lab = meta[:,0]
            y_theta   = meta[:,1:].view(-1, 2, 3)            

            if self.cuda:
                x_org   = x_org.cuda()
                x_img   = x_img.cuda() 
                y_mask  = y_mask.cuda() 
                y_lab   = y_lab.cuda()
                y_theta = y_theta.cuda()
            
            # fit (forward)            
            z, y_lab_hat, att, _, _ = self.net( x_img, x_img*y_mask[:,1,...].unsqueeze(dim=1) )                
            
            # measure accuracy and record loss           
            loss_bce  = self.criterion_bce(  y_lab_hat, y_lab.long() )
            loss_gmm  = self.criterion_gmm(  z, y_lab )              
            loss_att  = self.criterion_att(  x_org, y_mask, att )            
            loss      = loss_bce + loss_gmm + loss_att           
            topk      = self.topk( y_lab_hat, y_lab.long() )
            gmm       = self.gmm( z, y_lab )            
            
            # optimizer
            self.optimizer.zero_grad()
            (loss).backward() #batch_size
            self.optimizer.step()
            
            # update
            self.logger_train.update(
                {'loss': loss.cpu().item(), 'loss_gmm': loss_gmm.cpu().item(), 'loss_bce': loss_bce.cpu().item(), 'loss_att':loss_att.cpu().item() },
                {'topk': topk[0][0].cpu(), 'gmm': gmm.cpu().item() },
                batch_size,
                )
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.print_freq == 0:  
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )


    def evaluate(self, data_loader, epoch=0):
        
        # reset loader
        self.logger_val.reset()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, (x_org, x_img, y_mask, meta) in enumerate(data_loader):
                
                # get data (image, label)
                batch_size = x_img.shape[0]    
                                               
                y_lab = meta[:,0]
                y_theta   = meta[:,1:].view(-1, 2, 3)
                                
                if self.cuda:
                    x_org   = x_org.cuda()
                    x_img   = x_img.cuda()
                    y_mask  = y_mask.cuda()
                    y_lab   = y_lab.cuda()
                    y_theta = y_theta.cuda()
                
                # fit (forward)            
                z, y_lab_hat, att, fmap, srf  = self.net( x_img )                 
                
                # measure accuracy and record loss       
                loss_bce  = self.criterion_bce( y_lab_hat, y_lab.long() )
                loss_gmm  = self.criterion_gmm( z, y_lab )
                loss_att  = self.criterion_att( x_org, y_mask, att  )
                loss      = loss_bce + loss_gmm + loss_att           
                topk      = self.topk( y_lab_hat, y_lab.long() )               
                gmm       = self.gmm( z, y_lab )
                                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                                
                # update
                self.logger_val.update( 
                    {'loss': loss.cpu().item(), 'loss_gmm': loss_gmm.cpu().item(), 'loss_bce': loss_bce.cpu().item(), 'loss_att':loss_att.cpu().item() },
                    {'topk': topk[0][0].cpu(), 'gmm': gmm.cpu().item() },    
                    batch_size,          
                    )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader), 
                        batch_time, 
                        bplotter=False,
                        bavg=True, 
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['topk'].avg

        self.logger_val.logger(
            epoch, epoch, i, len(data_loader), 
            batch_time,
            bplotter=True,
            bavg=True, 
            bsummary=True,
            )

        #vizual_freq
        if epoch % self.view_freq == 0:

            att   = att[0,:,:,:].permute( 1,2,0 ).mean(dim=2)
            srf   = srf[0,:,:,:].permute( 1,2,0 ).sum(dim=2)  
            fmap  = fmap[0,:,:,:].permute( 1,2,0 ) 
                                    
            self.visheatmap.show('Image', x_img.data.cpu()[0].numpy()[0,:,:])           
            self.visheatmap.show('Image Attention',att.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Feature Map',srf.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Attention Map',fmap.cpu().numpy().astype(np.float32) )
            
        return acc

    
    def representation( self, dataloader, breal=True ):
        Y_labs = []
        Y_lab_hats = []
        Zs = []         
        self.net.eval()
        with torch.no_grad():
            for i_batch, sample in enumerate( tqdm(dataloader) ):

                if breal:                 
                    x_img, y_lab = sample['image'], sample['label']
                    y_lab = y_lab.argmax(dim=1)
                else:
                    x_org, x_img, y_mask, y_lab = sample
                    y_lab=y_lab[:,0]

                if self.cuda:
                    x_img = x_img.cuda()

                z, y_lab_hat, _,_,_ = self.net( x_img )
                Y_labs.append(y_lab)
                Y_lab_hats.append(y_lab_hat.data.cpu())
                Zs.append(z.data.cpu())
                
        Y_labs = np.concatenate( Y_labs, axis=0 )
        Y_lab_hats = np.concatenate( Y_lab_hats, axis=0 )
        Zs = np.concatenate( Zs, axis=0 )        
        return Y_labs, Y_lab_hats, Zs
    
    
    def __call__(self, image ):        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image    
            z, y_lab_hat, att, fmap, srf = self.net(x)                         
            y_lab_hat = F.softmax( y_lab_hat, dim=1 )            
        return z, y_lab_hat, att, fmap, srf


    def _create_loss(self, loss):

        # create loss
        if loss == 'attloss':            
            self.criterion_bce = nn.CrossEntropyLoss().cuda()
            self.criterion_att = nloss.Attloss()
            self.criterion_gmm = nloss.DGMMLoss( self.num_classes, cuda=self.cuda )            
        else:
            assert(False)

        self.s_loss = loss



class AttentionGMMSTNNeuralNet(AttentionNeuralNetAbstract):
    """
    Attention Neural Net and GMM representation with STN
    Args:
        -patchproject (str): path project
        -nameproject (str):  name project
        -no_cuda (bool): system cuda (default is True)
        -parallel (bool)
        -seed (int)
        -print_freq (int)
        -gpu (int)
        -view_freq (in epochs)
    """
    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0,
        view_freq=1
        ):
        super(AttentionGMMSTNNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu, view_freq  )
        

 
    def create(self, 
        arch, 
        num_output_channels,
        num_input_channels,        
        loss,
        lr,
        optimizer,
        lrsch,
        momentum=0.9,
        weight_decay=5e-4,
        pretrained=False,
        size_input=388,
        num_classes=8,
        ):
        """
        Create    
        Args:        
            -arch (string): architecture
            -num_output_channels,
            -num_input_channels,  
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) : 
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
            -
        """        
        super(AttentionGMMSTNNeuralNet, self).create( 
            arch, 
            num_output_channels,
            num_input_channels,        
            loss,
            lr,
            optimizer,
            lrsch,
            momentum,
            weight_decay,
            pretrained,
            size_input,
            num_classes,          
        )

        self.logger_train = Logger( 'Train', ['loss', 'loss_gmm', 'loss_bce', 'loss_att', 'loss_stn' ], [ 'topk', 'gmm'], self.plotter  )
        self.logger_val   = Logger( 'Val  ', ['loss', 'loss_gmm', 'loss_bce', 'loss_att', 'loss_stn' ], [ 'topk', 'gmm'], self.plotter )
        
      
    def training(self, data_loader, epoch=0):        

        #reset logger
        self.logger_train.reset()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, (x_org, x_img, y_mask, meta ) in enumerate(data_loader):
            
            # measure data loading time
            data_time.update(time.time() - end)
            batch_size = x_img.shape[0]
            
            y_lab = meta[:,0]
            y_theta   = meta[:,1:].view(-1, 2, 3)            

            if self.cuda:
                x_org   = x_org.cuda()
                x_img   = x_img.cuda() 
                y_mask  = y_mask.cuda() 
                y_lab   = y_lab.cuda()
                y_theta = y_theta.cuda()
            
            # fit (forward)            
            z, y_lab_hat, att, theta, _, _, _ = self.net( x_img, x_img*y_mask[:,1,...].unsqueeze(dim=1) )                
            
            # measure accuracy and record loss           
            loss_bce  = self.criterion_bce(  y_lab_hat, y_lab.long() )
            loss_gmm  = self.criterion_gmm(  z, y_lab )              
            loss_att  = self.criterion_att( x_org, y_mask, att )
            loss_stn  = self.criterion_stn( x_org, y_theta, theta )            
            loss      = loss_bce + loss_gmm + loss_att + loss_stn          
            topk      = self.topk( y_lab_hat, y_lab.long() )
            gmm       = self.gmm( z, y_lab )            
            
            # optimizer
            self.optimizer.zero_grad()
            (loss*batch_size).backward()
            self.optimizer.step()
            
            # update
            self.logger_train.update(
                {'loss': loss.cpu().item(), 
                 'loss_gmm': loss_gmm.cpu().item(), 
                 'loss_bce': loss_bce.cpu().item(), 
                 'loss_att':loss_att.cpu().item(), 
                 'loss_stn':loss_stn.cpu().item() 
                },
                {'topk': topk[0][0].cpu(), 'gmm': gmm.cpu().item() },
                batch_size,
                )
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.print_freq == 0:  
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )


    def evaluate(self, data_loader, epoch=0):
        
        # reset loader
        self.logger_val.reset()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, (x_org, x_img, y_mask, meta) in enumerate(data_loader):
                
                # get data (image, label)
                batch_size = x_img.shape[0]    
                                               
                y_lab = meta[:,0]
                y_theta   = meta[:,1:].view(-1, 2, 3)
                                
                if self.cuda:
                    x_org   = x_org.cuda()
                    x_img   = x_img.cuda()
                    y_mask  = y_mask.cuda()
                    y_lab   = y_lab.cuda()
                    y_theta = y_theta.cuda()
                              

                # fit (forward)            
                z, y_lab_hat, att, theta, att_t, fmap, srf  = self.net( x_img ) 
                
                
                # measure accuracy and record loss       
                loss_bce  = self.criterion_bce(  y_lab_hat, y_lab.long() )
                loss_gmm  = self.criterion_gmm(  z, y_lab )
                loss_att  = self.criterion_att( x_org, y_mask, att )
                loss_stn  = self.criterion_stn( x_org, y_theta, theta ) 
                loss      = loss_bce + loss_gmm + loss_att + loss_stn          
                topk      = self.topk( y_lab_hat, y_lab.long() )               
                gmm       = self.gmm( z, y_lab )
                                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                                
                # update
                self.logger_val.update( 
                    {'loss': loss.cpu().item(), 
                     'loss_gmm': loss_gmm.cpu().item(), 
                     'loss_bce': loss_bce.cpu().item(), 
                     'loss_att':loss_att.cpu().item(), 
                     'loss_stn':loss_stn.cpu().item() 
                    },
                    {'topk': topk[0][0].cpu(), 'gmm': gmm.cpu().item() },    
                    batch_size,          
                    )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader), 
                        batch_time, 
                        bplotter=False,
                        bavg=True, 
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['topk'].avg

        self.logger_val.logger(
            epoch, epoch, i, len(data_loader), 
            batch_time,
            bplotter=True,
            bavg=True, 
            bsummary=True,
            )

        #vizual_freq
        if epoch % self.view_freq == 0:

            att   = att[0,:,:,:].permute( 1,2,0 ).mean(dim=2)
            att_t = att_t[0,:,:,:].permute( 1,2,0 ).mean(dim=2)
            srf   = srf[0,:,:,:].permute( 1,2,0 ).sum(dim=2)  
            fmap  = fmap[0,:,:,:].permute( 1,2,0 ) 
            
            print('theta')
            print(y_theta[0,:,:] )
            print(theta[0,:,:] )
                        
            self.visheatmap.show('Image', x_img.data.cpu()[0].numpy()[0,:,:])           
            self.visheatmap.show('Image Attention',att.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Image Attention Trans',att_t.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Feature Map',srf.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Attention Map',fmap.cpu().numpy().astype(np.float32) )
            
        return acc

    
    def representation( self, dataloader, breal=True ):
        Y_labs = []
        Y_lab_hats = []
        Zs = []         
        self.net.eval()
        with torch.no_grad():
            for i_batch, sample in enumerate( tqdm(dataloader) ):

                if breal:                 
                    x_img, y_lab = sample['image'], sample['label']
                    y_lab = y_lab.argmax(dim=1)
                else:
                    x_org, x_img, y_mask, y_lab = sample
                    y_lab=y_lab[:,0]

                if self.cuda:
                    x_img = x_img.cuda()

                z, y_lab_hat, _,_,_,_,_ = self.net( x_img )
                Y_labs.append(y_lab)
                Y_lab_hats.append(y_lab_hat.data.cpu())
                Zs.append(z.data.cpu())
                
        Y_labs = np.concatenate( Y_labs, axis=0 )
        Y_lab_hats = np.concatenate( Y_lab_hats, axis=0 )
        Zs = np.concatenate( Zs, axis=0 )        
        return Y_labs, Y_lab_hats, Zs
    
    
    def __call__(self, image ):        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image    
            z, y_lab_hat, att, theta, att_t, fmap, srf = self.net(x)                         
            y_lab_hat = F.softmax( y_lab_hat, dim=1 )            
        return z, y_lab_hat, att, theta, att_t, fmap, srf


    def _create_loss(self, loss):

        # create loss
        if loss == 'attloss':            
            self.criterion_bce = nn.CrossEntropyLoss().cuda()
            self.criterion_att = nloss.Attloss()
            self.criterion_stn = nloss.STNloss()
            self.criterion_gmm = nloss.DGMMLoss( self.num_classes, cuda=self.cuda )
        else:
            assert(False)

        self.s_loss = loss




class MitosisAttentionGMMNeuralNet(AttentionNeuralNetAbstract):
    """
    Mitosis Attention Neural Net and GMM representation 
    Args:
        -patchproject (str): path project
        -nameproject (str):  name project
        -no_cuda (bool): system cuda (default is True)
        -parallel (bool)
        -seed (int)
        -print_freq (int)
        -gpu (int)
        -view_freq (in epochs)
    """
    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0,
        view_freq=1
        ):
        super(MitosisAttentionGMMNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu, view_freq  )
        

 
    def create(self, 
        arch, 
        num_output_channels,
        num_input_channels,        
        loss,
        lr,
        optimizer,
        lrsch,
        momentum=0.9,
        weight_decay=5e-4,
        pretrained=False,
        size_input=388,
        num_classes=8,
        ):
        """
        Create    
        Args:        
            -arch (string): architecture
            -num_output_channels,
            -num_input_channels,  
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) : 
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
            -
        """        
        super(MitosisAttentionGMMNeuralNet, self).create( 
            arch, 
            num_output_channels,
            num_input_channels,        
            loss,
            lr,
            optimizer,
            lrsch,
            momentum,
            weight_decay,
            pretrained,
            size_input,
            num_classes,          
        )

        self.logger_train = Logger( 'Train', ['loss', 'loss_gmm', 'loss_bce', 'loss_att' ], [ 'topk', 'gmm'], self.plotter  )
        self.logger_val   = Logger( 'Val  ', ['loss', 'loss_gmm', 'loss_bce', 'loss_att' ], [ 'topk', 'gmm'], self.plotter )
        
      
    def training(self, data_loader, epoch=0):        

        #reset logger
        self.logger_train.reset()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, (x_org, x_img, y_mask, meta ) in enumerate(data_loader):
            
            # measure data loading time
            data_time.update(time.time() - end)
            batch_size = x_img.shape[0]
            
            y_lab = meta[:,0]
            y_reg = meta[:,1]
            y_theta = meta[:,2:].view(-1, 2, 3)           

            if self.cuda:
                x_org   = x_org.cuda()
                x_img   = x_img.cuda() 
                y_mask  = y_mask.cuda() 
                y_lab   = y_lab.cuda()
                y_reg   = y_reg.cuda()
                y_theta = y_theta.cuda()
            
            # fit (forward)            
            z, y_lab_hat, att, _, _ = self.net( x_img, x_img*y_mask[:,1,...].unsqueeze(dim=1) )                
            
            # measure accuracy and record loss           
            loss_bce  = self.criterion_bce(  y_lab_hat, y_lab.long() )
            loss_gmm  = self.criterion_gmm( z, y_reg, data_loader.dataset.numclass_reg  )
            loss_att  = self.criterion_att(  x_org, y_mask, att )            
            loss      = loss_bce + loss_gmm + loss_att           
            topk      = self.topk( y_lab_hat, y_lab.long() )
            gmm       = self.gmm( z, y_lab, data_loader.dataset.numclass_reg )            
            
            # optimizer
            self.optimizer.zero_grad()
            (loss).backward() #batch_size
            self.optimizer.step()
            
            # update
            self.logger_train.update(
                {'loss': loss.cpu().item(), 'loss_gmm': loss_gmm.cpu().item(), 'loss_bce': loss_bce.cpu().item(), 'loss_att':loss_att.cpu().item() },
                {'topk': topk[0][0].cpu(), 'gmm': gmm.cpu().item() },
                batch_size,
                )
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.print_freq == 0:  
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )


    def evaluate(self, data_loader, epoch=0):
        
        # reset loader
        self.logger_val.reset()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, (x_org, x_img, y_mask, meta) in enumerate(data_loader):
                
                # get data (image, label)
                batch_size = x_img.shape[0]    
                                               
                y_lab = meta[:,0]
                y_reg = meta[:,1]
                y_theta   = meta[:,2:].view(-1, 2, 3)
                                
                if self.cuda:
                    x_org   = x_org.cuda()
                    x_img   = x_img.cuda()
                    y_mask  = y_mask.cuda()
                    y_lab   = y_lab.cuda()
                    y_reg   = y_reg.cuda()
                    y_theta = y_theta.cuda()
                
                # fit (forward)            
                z, y_lab_hat, att, fmap, srf  = self.net( x_img )                 
                
                # measure accuracy and record loss       
                loss_bce  = self.criterion_bce( y_lab_hat, y_lab.long() )
                loss_gmm  = self.criterion_gmm( z, y_reg, data_loader.dataset.numclass_reg )
                loss_att  = self.criterion_att( x_org, y_mask, att  )
                loss      = loss_bce + loss_gmm + loss_att           
                topk      = self.topk( y_lab_hat, y_lab.long() )               
                gmm       = self.gmm( z, y_lab, data_loader.dataset.numclass_reg )
                                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                                
                # update
                self.logger_val.update( 
                    {'loss': loss.cpu().item(), 'loss_gmm': loss_gmm.cpu().item(), 'loss_bce': loss_bce.cpu().item(), 'loss_att':loss_att.cpu().item() },
                    {'topk': topk[0][0].cpu(), 'gmm': gmm.cpu().item() },    
                    batch_size,          
                    )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader), 
                        batch_time, 
                        bplotter=False,
                        bavg=True, 
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['topk'].avg

        self.logger_val.logger(
            epoch, epoch, i, len(data_loader), 
            batch_time,
            bplotter=True,
            bavg=True, 
            bsummary=True,
            )

        #vizual_freq
        if epoch % self.view_freq == 0:

            att   = att[0,:,:,:].permute( 1,2,0 ).mean(dim=2)
            srf   = srf[0,:,:,:].permute( 1,2,0 ).sum(dim=2)  
            fmap  = fmap[0,:,:,:].permute( 1,2,0 ).mean(dim=2) 
                                    
            self.visheatmap.show('Image', x_img.data.cpu()[0].numpy()[0,:,:])           
            self.visheatmap.show('Image Attention',att.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Feature Map',srf.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Attention Map',fmap.cpu().numpy().astype(np.float32) )
            
        return acc

    
    def representation( self, dataloader, breal=True ):
        
        Y_labs = []
        Y_regs = []
        Y_lab_hats = []
        Zs = []        
        
        self.net.eval()
        with torch.no_grad():
            for i_batch, sample in enumerate( tqdm(dataloader) ):

                if breal:                 
                    x_img, y_lab = sample['image'], sample['label']
                    y_lab = y_lab.argmax(dim=1)
                else:
                    x_org, x_img, y_mask, meta = sample                    
                    y_lab=meta[:,0]
                    y_reg=meta[:,1]

                if self.cuda:
                    x_img = x_img.cuda()

                z, y_lab_hat, _, _, _ = self.net( x_img )
                
                Y_labs.append(y_lab)
                Y_regs.append(y_reg)
                Y_lab_hats.append(y_lab_hat.data.cpu())
                Zs.append(z.data.cpu())
                
        Y_labs = np.concatenate( Y_labs, axis=0 )
        Y_regs = np.concatenate( Y_regs, axis=0 )
        Y_lab_hats = np.concatenate( Y_lab_hats, axis=0 )
        Zs = np.concatenate( Zs, axis=0 )  
        
        return Y_labs, Y_regs, Y_lab_hats, Zs
    
    
    def __call__(self, image ):        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image    
            z, y_lab_hat, att, fmap, srf = self.net(x)                         
            y_lab_hat = F.softmax( y_lab_hat, dim=1 )            
        return z, y_lab_hat, att, fmap, srf


    def _create_loss(self, loss):

        # create loss
        if loss == 'attloss':            
            self.criterion_bce = nn.CrossEntropyLoss().cuda()
            self.criterion_att = nloss.Attloss()
            self.criterion_gmm = nloss.DGMMLoss( self.num_classes, cuda=self.cuda )            
        else:
            assert(False)

        self.s_loss = loss
