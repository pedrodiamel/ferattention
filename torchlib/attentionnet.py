

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


class AttentionNeuralNet(NeuralNetAbstract):
    """
    Attention Neural Net 
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

        super(AttentionNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu  )
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
        cfg_scheduler={ 'step_size':30, 'gamma':0.1  }        
        self.num_classes = num_classes
        
        super(AttentionNeuralNet, self).create( 
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
        self.gmm      = nloss.GMMAccuracy( classes=num_classes  )
        self.dice     = nloss.Dice()
       
        # Set the graphic visualization
        self.logger_train = Logger( 'Train', ['loss', 'loss_gmm', 'loss_bce', 'loss_att' ], [ 'topk', 'gmm'], self.plotter  )
        self.logger_val   = Logger( 'Val  ', ['loss', 'loss_gmm', 'loss_bce', 'loss_att' ], [ 'topk', 'gmm'], self.plotter )

        self.visheatmap = gph.HeatMapVisdom(env_name=self.nameproject, heatsize=(100,100) )
        self.visimshow  = gph.ImageVisdom(env_name=self.nameproject, imsize=(100,100) )

      
    def training(self, data_loader, epoch=0):        

        #reset logger
        self.logger_train.reset()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, (x_org, x_img, y_mask, y_lab ) in enumerate(data_loader):
            
            # measure data loading time
            data_time.update(time.time() - end)
            batch_size = x_img.shape[0]
            y_lab = y_lab.squeeze(dim=1)

            if self.cuda:
                x_org   = x_org.cuda()
                x_img   = x_img.cuda() 
                y_mask  = y_mask.cuda() 
                y_lab   = y_lab.cuda()

            # fit (forward)            
            z, y_lab_hat, att, _, _ = self.net( x_img )                

            # measure accuracy and record loss           
            loss_bce  = self.criterion_bce(  y_lab_hat, y_lab.long() )
            loss_gmm  = self.criterion_gmm(  z, y_lab )            
            loss_att =  ((( (x_org*y_mask[:,1,...].unsqueeze(dim=1)) - att ) ** 2)).mean()
            loss_att = torch.clamp(loss_att, max=10)
            
            loss = loss_bce + loss_gmm + loss_att           
            
            topk  = self.topk( y_lab_hat, y_lab.long() )
            gmm  = self.gmm( z, y_lab )            
            
            # optimizer
            self.optimizer.zero_grad()
            (loss).backward() #batch_size
            self.optimizer.step()
            
            # update
            self.logger_train.update(
                {'loss': loss.data[0], 'loss_gmm': loss_gmm.data[0], 'loss_bce': loss_bce.data[0], 'loss_att':loss_att.data[0] },
                {'topk': topk[0][0], 'gmm': gmm.data[0] },
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
            for i, (x_org, x_img, y_mask, y_lab) in enumerate(data_loader):
                
                # get data (image, label)
                batch_size = x_img.shape[0]                
                y_lab = y_lab.squeeze(dim=1)
                
                
                if self.cuda:
                    x_org  = x_org.cuda()
                    x_img  = x_img.cuda()
                    y_mask = y_mask.cuda()
                    y_lab  = y_lab.cuda()
                              

                # fit (forward)            
                z, y_lab_hat, att, fmap, srf  = self.net( x_img ) 
                
                
                # measure accuracy and record loss       
                loss_bce  = self.criterion_bce(  y_lab_hat, y_lab.long() )
                loss_gmm  = self.criterion_gmm(  z, y_lab )
                loss_att   =  ((( (x_org*y_mask[:,1,...].unsqueeze(dim=1)) - att ) ** 2)).mean()  
                loss_att = torch.clamp(loss_att, max=10)
                
                loss      = loss_bce + loss_gmm + loss_att           

                topk  = self.topk( y_lab_hat, y_lab.long() )               
                gmm  = self.gmm( z, y_lab )
                
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                               
                
                # update
                self.logger_val.update( 
                    {'loss': loss.data[0], 'loss_gmm': loss_gmm.data[0], 'loss_bce': loss_bce.data[0], 'loss_att':loss_att.data[0] },
                    {'topk': topk[0][0], 'gmm': gmm.data[0] },    
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

            att = att[0,:,:,:].permute( 1,2,0 ).mean(dim=2)  
            srf = srf[0,:,:,:].permute( 1,2,0 ).sum(dim=2)  
            fmap = fmap[0,:,:,:].permute( 1,2,0 ) 
                        
            self.visheatmap.show('Image', x_img.data.cpu()[0].numpy()[0,:,:])           
            self.visheatmap.show('Image Attention',att.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Feature Map',srf.cpu().numpy().astype(np.float32) )
            self.visheatmap.show('Attention Map',fmap.cpu().numpy().astype(np.float32) )
            

        return acc


    def test(self, data_loader ):        
        # initialization
        k, masks, ids = 0, [], []
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, (idd, x_img) in enumerate( tqdm(data_loader) ):                
                x = x_img.cuda() if self.cuda else x_img                    
                # fit (forward)
                y_hat = self.net(x)
                y_hat = F.softmax(y_hat, dim=1)    
                y_hat = pytutils.to_np(y_hat)
                masks.append( y_hat )
                ids.append( idd )      
        return ids, masks

    
    def __call__(self, image ):        
        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image    
            z, y_lab_hat, att, fmap, srf = self.net(x)                         
            y_lab_hat = F.softmax( y_lab_hat, dim=1 )
            
        return z, y_lab_hat, att, fmap, srf


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
            
        

    def _create_loss(self, loss):

        # create loss
        if loss == 'attgmm':            
            self.criterion_bce = nn.CrossEntropyLoss().cuda()
            self.criterion_mse = nn.MSELoss().cuda()
            self.criterion_gmm = nloss.DGMMLoss( self.num_classes, cuda=self.cuda )
        else:
            assert(False)

        self.s_loss = loss




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

