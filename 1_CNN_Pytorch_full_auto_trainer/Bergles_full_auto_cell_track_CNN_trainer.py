# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================
@author: Tiger


Add:
        - loss plot only accumulated loss
        - validation over 1000 images per epoch
        - smaller kernel size
        - smaller image crop
        - RAM???
"""
""" Install for Pytorch:
     
     1) must first install cuda
     2) conda install pytorch torchvision cudatoolkit=10.2 -c pytorch   # Must indicate cuda version???
     
     3) pip install torchio
    
     torch.__version__
     

     *** must add device(device) ==> to send things to GPU!

     
     ***crazy speed up with line below
     torch.backends.cudnn.benchmark = True
     
     
     torch.backends.cudnn.deterministic=True

     ***For uninstall:
     conda install pytorch torchvision cudatoolkit=10.2 -c pytorch  
     
     
     put import statements at top of file or may crash???
     
     Add main function so can do multi-threading?
     no classes within main function
     
   
     use volatile == True for inference mode???
"""

""" ALLOWS print out of results on compute canada """
import matplotlib
matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8) 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

""" Libraries to load """
import numpy as np
from plot_functions_CLEANED import *
from data_functions_CLEANED import *
from data_functions_3D import *
import glob, os
import datetime
import time
import bcolz
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from UNet_pytorch import *
from UNet_pytorch_online import *
from PYTORCH_dataloader import *

from sklearn.model_selection import train_test_split

from unet_nested import *
import kornia


torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
        
    """ Define GPU to use """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    """" Input paths """    
    s_path = './(1) Checkpoints_full_auto_no_spatialW/'
    s_path = './(2) Checkpoints_full_auto_spatialW/'
    
    
    s_path = './(3) Checkpoints_full_auto_no_spatialW_nested_unet/'
    
    input_path = '/media/user/storage/Data/(2) cell tracking project/a_training_data_GENERATE_FULL_AUTO/Training_cell_track_full_auto_COMPLETED/'

    resume = 0
    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
    onlyfiles_check.sort(key = natsort_key1)
    
    
    # """ load mean and std """  
    mean_arr = np.load('./normalize_pytorch_CLEANED/mean_VERIFIED.npy')
    std_arr = np.load('./normalize_pytorch_CLEANED/std_VERIFIED.npy')       


    num_workers = 2;
    
    
    save_every_num_epochs = 1;
    plot_every_num_epochs = 1;
    validate_every_num_epochs = 1;  
    
    deep_supervision = False;
    
    
    if not onlyfiles_check:   
        """ Get metrics per batch """
        train_loss_per_batch = []; train_jacc_per_batch = []
        val_loss_per_batch = []; val_jacc_per_batch = []
        """ Get metrics per epoch"""
        train_loss_per_epoch = []; train_jacc_per_epoch = []
        val_loss_per_eval = []; val_jacc_per_eval = []
        plot_sens = []; plot_sens_val = []; plot_acc = []
        plot_prec = []; plot_prec_val = [];
        lr_plot = [];
        iterations = 0;
        
        """ Start network """           
        # unet = UNet_online(in_channels=4, n_classes=2, depth=5, wf=3, padding= int((5 - 1)/2), 
        #                    batch_norm=True, batch_norm_switchable=False, up_mode='upconv')


        kernel_size = 5
        pad = int((kernel_size - 1)/2)
        #unet = UNet(in_channel=1,out_channel=2, kernel_size=kernel_size, pad=pad)
        
        #kernel_size = 5
        #unet = UNet_online(in_channels=2, n_classes=2, depth=5, wf=3, kernel_size = kernel_size, padding= int((kernel_size - 1)/2), 
        #                    batch_norm=True, batch_norm_switchable=False, up_mode='upsample')


        unet = NestedUNet(num_classes=2, input_channels=4, deep_supervision=deep_supervision, padding=pad, batch_norm_switchable=False)

        unet.train()
        unet.to(device)
        print('parameters:', sum(param.numel() for param in unet.parameters()))  
    
        """ Select loss function """
        loss_function = torch.nn.CrossEntropyLoss(reduction='none')

        """ Select optimizer """
        lr = 1e-5; milestones = [100]  # with AdamW slow down

        optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

        """ Add scheduler """
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
        resume = 0
        
        """ Prints out all variables in current graph """
        # Required to initialize all
        batch_size = 4;       
        test_size = 0.1
        """ Load training data """
        print('loading data')   
        """ Specify transforms """
        #transforms = initialize_transforms(p=0.5)
        #transforms = initialize_transforms_simple(p=0.5)
        transforms = 0
        
        sp_weight_bool = 0
        
    else:             
        """ Find last checkpoint """       
        last_file = onlyfiles_check[-1]
        split = last_file.split('check_')[-1]
        num_check = split.split('.')
        checkpoint = num_check[0]
        checkpoint = 'check_' + checkpoint

        print('restoring weights')
        check = torch.load(s_path + checkpoint)
        cur_epoch = check['cur_epoch']
        iterations = check['iterations']
        idx_train = check['idx_train']
        idx_valid = check['idx_valid']
        
        
        unet = check['model_type']
        optimizer = check['optimizer_type']
        scheduler = check['scheduler_type']
        
        
        unet.load_state_dict(check['model_state_dict'])
        unet.to(device)
        optimizer.load_state_dict(check['optimizer_state_dict'])
        scheduler.load_state_dict(check['scheduler'])
        
        """ Restore per batch """
        train_loss_per_batch = check['train_loss_per_batch']
        train_jacc_per_batch = check['train_jacc_per_batch']
        val_loss_per_batch = check['val_loss_per_batch']
        val_jacc_per_batch = check['val_jacc_per_batch']
     
        """ Restore per epoch """
        train_loss_per_epoch = check['train_loss_per_epoch']
        train_jacc_per_epoch = check['train_jacc_per_epoch']
        val_loss_per_eval = check['val_loss_per_eval']
        val_jacc_per_eval = check['val_jacc_per_eval']
     
        plot_sens = check['plot_sens']
        plot_sens_val = check['plot_sens_val']
        plot_prec = check['plot_prec']
        plot_prec_val = check['plot_prec_val']
        plot_acc = check['plot_acc']
     
        lr_plot = check['lr_plot']
        
        
        # newly added
        mean_arr = check['mean_arr']
        std_arr = check['std_arr']
        
        batch_size = check['batch_size']
        sp_weight_bool = check['sp_weight_bool']
        loss_function = check['loss_function']
        transforms = check['transforms']

        resume = 1
    #transforms = initialize_transforms_simple(p=0.5)

    """ 
        Time for 50 iterations (batch size 2)
        411.34 ==> 0 workers
        183.78438819999997 ==> 2 workers 
        127 ==> 4 workers ==> 60 secs training generator time
        111.03402260000001 ==> no transform
        109 ==> 6 workers ==> 90 secs training generator time   
        110.0 ==> 6 workers WITH DIRECT LOADING FROM BCOLZ ==> 30 secs generator tim
        max is 1000 samples loaded with batch size 2 and even with batch size 1??? 
        
        
        On Titan:
            time for 50 batch size 2 ==> 71 secs
            time for 50 batch size 4 ==> 116 secs (with transforms)
        
    """ 

    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*_crop_input_cur.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,seed          = i.replace('_crop_input_cur.tif','_crop_input_cur_seed.tif'),
                             cur_full_seg  = i.replace('_crop_input_cur.tif','_crop_input_cur_seg_FULL.tif'),
                             next_input    = i.replace('_crop_input_cur.tif','_crop_input_next.tif'),
                             next_full_seg = i.replace('_crop_input_cur.tif','_crop_input_next_seg_FULL.tif'),
                             truth         = i.replace('_crop_input_cur.tif','_crop_truth.tif')) for i in images]
    counter = list(range(len(examples)))
    
    if not resume:
        idx_train, idx_valid, empty, empty = train_test_split(counter, counter, test_size=test_size, random_state=2018)
        
    training_set = Dataset_tiffs(idx_train, examples, mean_arr, std_arr, sp_weight_bool=sp_weight_bool, transforms = transforms)
    val_set = Dataset_tiffs(idx_valid, examples, mean_arr, std_arr, sp_weight_bool=sp_weight_bool, transforms = 0)
    
    """ Create training and validation generators"""
    val_generator = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True)

    training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, sampler=None,
                      batch_sampler=None, num_workers=num_workers, collate_fn=None,
                      pin_memory=True, drop_last=False, timeout=0,
                      worker_init_fn=None)
         

    print('Total # training images per epoch: ' + str(len(training_set)))
    print('Total # validation images: ' + str(len(val_set)))

    """ Epoch info """
    train_steps_per_epoch = len(idx_train)/batch_size
    validation_size = len(idx_valid)
    epoch_size = len(idx_train)    


    """ Start training """
    starter = 0
    for cur_epoch in range(len(train_loss_per_epoch), 10000):   
         unet.train        
         loss_train = 0
         jacc_train = 0   
                  
         for param_group in optimizer.param_groups:
              #param_group['lr'] = 1e-6   # manually sets learning rate
              cur_lr = param_group['lr']
              lr_plot.append(cur_lr)
              print('Current learning rate is: ' + str(cur_lr))
              
              
         iter_cur_epoch = 0;          
         for batch_x, batch_y, spatial_weight in training_generator:
                starter += 1
                if starter == 1:
                    start = time.perf_counter()
                if starter == 50:
                    stop = time.perf_counter(); diff = stop - start; print(diff)
                    
                    
                # PRINT OUT THE SHAPE OF THE INPUT
                if iter_cur_epoch == 0:
                    print('input size is' + str(batch_x.shape))
                    
                    
                """ Plot for debug """    
                # np_inputs = np.asarray(batch_x.numpy()[0], dtype=np.uint8)
                # np_labels = np.asarray(batch_y.numpy()[0], dtype=np.uint8)
                # np_labels[np_labels > 0] = 255
                
                # imsave(s_path + str(iterations) + '_input.tif', np_inputs)
                # imsave(s_path + str(iterations) + '_label.tif', np_labels)
                
                # in_max = plot_max(np_inputs, plot=0)
                # lb_max = plot_max(np_labels, plot=0)
                
                
                # imsave(s_path + str(iterations) + '_max_input.tif', in_max)
                # imsave(s_path + str(iterations) + '_max_label.tif', lb_max)                
                
                """ Load data ==> shape is (batch_size, num_channels, depth, height, width)
                     (1) converts to Tensor
                     (2) normalizes + appl other transforms on GPU
                     (3) ***add non-blocking???
                     ***INPUT LABELS MUST BE < 255??? or else get CudNN error
                     
                """
                inputs, labels = transfer_to_GPU(batch_x, batch_y, device, mean_arr, std_arr)
   
                """ zero the parameter gradients"""
                optimizer.zero_grad()       
                
                """ forward + backward + optimize """
                output_train = unet(inputs)
                
                loss = loss_function(output_train, labels)
                if torch.is_tensor(spatial_weight):
                     spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)          
                     weighted = loss * spatial_tensor
                     loss = torch.mean(weighted)
                else:
                     loss = torch.mean(loss)   
                     #loss
                
                
                loss.backward()
                optimizer.step()
               
                """ Training loss """
                """ ********************* figure out how to do spatial weighting??? """
                train_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                loss_train += loss.cpu().data.numpy()
   
                """ Calculate Jaccard on GPU """                 
                jacc = jacc_eval_GPU_torch(output_train, labels)
                jacc = jacc.cpu().data.numpy()
                                            
                jacc_train += jacc # Training jacc
                train_jacc_per_batch.append(jacc)
   
                iterations = iterations + 1       
                iter_cur_epoch += 1
              
                if iterations % 100 == 0:
                    print('Trained: %d' %(iterations))

               
                
    
         train_loss_per_epoch.append(loss_train/iter_cur_epoch)
         train_jacc_per_epoch.append(jacc_train/iter_cur_epoch)              
    
         """ Should I keep track of loss on every single sample? and iteration? Just not plot it??? """   
         loss_val = 0; jacc_val = 0
         precision_val = 0; sensitivity_val = 0; val_idx = 0;
         iter_cur_epoch = 0;
         
         total_FPs = 0; total_TPs = 0; total_FNs = 0; total_TNs = 0;
         if cur_epoch % validate_every_num_epochs == 0:
             
              with torch.set_grad_enabled(False):  # saves GPU RAM
                  unet.eval()
                  for batch_x_val, batch_y_val, spatial_weight in val_generator:
                        
                        """ Transfer to GPU to normalize ect... """
                        inputs_val, labels_val = transfer_to_GPU(batch_x_val, batch_y_val, device, mean_arr, std_arr)
             
                        # forward pass to check validation
                        output_val = unet(inputs_val)
                        if torch.is_tensor(spatial_weight):
                               spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)          
                               weighted = loss * spatial_tensor
                               loss = torch.mean(weighted)
                        else:
                               loss = torch.mean(loss)  
          
                        """ Training loss """
                        val_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                        loss_val += loss.cpu().data.numpy()
                                         
                        """ Calculate jaccard on GPU """
                        jacc = jacc_eval_GPU_torch(output_val, labels_val)
                        jacc = jacc.cpu().data.numpy()
                        
                        jacc_val += jacc
                        val_jacc_per_batch.append(jacc)

                        """ Convert back to cpu """                                      
                        output_val = output_val.cpu().data.numpy()            
                        output_val = np.moveaxis(output_val, 1, -1)
                        
                        """ Calculate # of TP, FP, FN, TN for every image in the batch"""
                        batch_y_val = batch_y_val.cpu().data.numpy() 
                        labels_val = labels_val.cpu().data.numpy()
                        for b_idx in range(len(batch_y_val)):
                            
                            # get image from current batch
                            seg_val = np.argmax(output_val[b_idx], axis=-1)  
                            cur_label = labels_val[b_idx]
                            
                            
                            ### (1) If is TN or FN, the seg_val is blank
                            if not np.count_nonzero(seg_val):
                                
                                ### It is a true negative if the seg_val is ALSO blank
                                if not np.count_nonzero(cur_label):
                                   total_TNs += 1
                                   
                                else:   
                                   ### otherwise, calculate number of FN's
                                   # seg_val[seg_val > 0] = 1
                                   # labelled = measure.label(seg_val)
                                   # cc_coloc = measure.regionprops(labelled)
                                   
                                   total_FNs += 1

                            
                            else:
                                ### (2) otherwise, test for TP and FP
                                only_coloc, TP, FP = find_overlap_objs(cur_label, seg_val)
                            
                                total_TPs += TP
                                total_FPs += FP
                                
                        #zzz
                        val_idx = val_idx + batch_size
                        print('Validation: ' + str(val_idx) + ' of total: ' + str(validation_size))
                
                        iter_cur_epoch += 1
                
              accuracy = (total_TPs + total_TNs)/(total_TPs + total_TNs + total_FPs + total_FNs)
              sensitivity = total_TPs/(total_TPs + total_FNs);
              precision = total_TPs/(total_TPs + total_FPs);     
                
              val_loss_per_eval.append(loss_val/iter_cur_epoch)
              val_jacc_per_eval.append(jacc_val/iter_cur_epoch)    
                   
              plot_prec.append(precision)
              plot_sens.append(sensitivity)
              
              plot_acc.append(accuracy)
                   
                  
              """ Add to scheduler to do LR decay """
              scheduler.step()
                  
         if cur_epoch % plot_every_num_epochs == 0:       
             
            
              """ Plot sens + precision + jaccard + loss """
              plot_metric_fun(plot_acc, plot_acc, class_name='', metric_name='accuracy', plot_num=29)
              plt.figure(29); plt.savefig(s_path + 'Accuracy.png')

              plot_metric_fun(plot_sens, plot_sens_val, class_name='', metric_name='sensitivity', plot_num=30)
              plt.figure(30); plt.savefig(s_path + 'Sensitivity.png')
                    
              plot_metric_fun(plot_prec, plot_prec_val, class_name='', metric_name='precision', plot_num=31)
              plt.figure(31); plt.savefig(s_path + 'Precision.png')
           
              plot_metric_fun(train_jacc_per_epoch, val_jacc_per_eval, class_name='', metric_name='jaccard', plot_num=32)
              plt.figure(32); plt.savefig(s_path + 'Jaccard.png')
                   
                 
              plot_metric_fun(train_loss_per_epoch, val_loss_per_eval, class_name='', metric_name='loss', plot_num=33)
              plt.figure(33); plt.yscale('log'); plt.savefig(s_path + 'loss_per_epoch.png')          
                   
              
              plot_metric_fun(lr_plot, [], class_name='', metric_name='learning rate', plot_num=35)
              plt.figure(35); plt.savefig(s_path + 'lr_per_epoch.png') 
     

              """ Plot metrics per batch """                
              plot_metric_fun(train_jacc_per_batch, [], class_name='', metric_name='jaccard', plot_num=34)
              plt.figure(34); plt.savefig(s_path + 'Jaccard_per_batch.png')
                                
              plot_cost_fun(train_loss_per_batch, train_loss_per_batch)                   
              plt.figure(18); plt.savefig(s_path + 'global_loss.png')
              plt.figure(19); plt.savefig(s_path + 'detailed_loss.png')
              plt.figure(25); plt.savefig(s_path + 'global_loss_LOG.png')
              plt.close('all')
                                 
              plot_depth = 8
              output_train = output_train.cpu().data.numpy()            
              output_train = np.moveaxis(output_train, 1, -1)              
              seg_train = np.argmax(output_train[0], axis=-1)  
              
              # convert back to CPU
              batch_x = batch_x.cpu().data.numpy() 
              batch_y = batch_y.cpu().data.numpy() 
              batch_x_val = batch_x_val.cpu().data.numpy()
              
              
              plot_trainer_3D_PYTORCH_cell_track_AUTO(seg_train, seg_val, batch_x[0, 0:2, ...], batch_x_val[0, 0:2, ...], batch_y[0], batch_y_val[0],
                                       s_path, iterations, plot_depth=plot_depth)
                                             
              
         """ To save (every x iterations) """
         if cur_epoch % save_every_num_epochs == 0:                          
               save_name = s_path + 'check_' +  str(iterations)               
               torch.save({
                'cur_epoch': cur_epoch,
                'iterations': iterations,
                'idx_train': idx_train,
                'idx_valid': idx_valid,
                
                
                'model_type': unet,
                'optimizer_type': optimizer,
                'scheduler_type': scheduler,
                
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                
                'train_loss_per_batch': train_loss_per_batch,
                'train_jacc_per_batch': train_jacc_per_batch,
                'val_loss_per_batch': val_loss_per_batch,
                'val_jacc_per_batch': val_jacc_per_batch,
                
                'train_loss_per_epoch': train_loss_per_epoch,
                'train_jacc_per_epoch': train_jacc_per_epoch,
                'val_loss_per_eval': val_loss_per_eval,
                'val_jacc_per_eval': val_jacc_per_eval,
                
                'plot_sens': plot_sens,
                'plot_sens_val': plot_sens_val,
                'plot_prec': plot_prec,
                'plot_prec_val': plot_prec_val,
                
                'plot_acc': plot_acc,
                
                'lr_plot': lr_plot,
                
                 # newly added
                'mean_arr': mean_arr,
                'std_arr': std_arr,
                
                'batch_size': batch_size,  
                'sp_weight_bool': sp_weight_bool,
                'loss_function': loss_function,  
                'transforms': transforms  

                
                
                }, save_name)
                

                
               
              
              