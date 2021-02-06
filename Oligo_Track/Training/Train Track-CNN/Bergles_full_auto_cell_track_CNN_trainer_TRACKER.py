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
from tracker import *
import glob, os
import datetime
import time
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from UNet_pytorch import *
from UNet_pytorch_online import *
from PYTORCH_dataloader import *
from HD_loss import *

from sklearn.model_selection import train_test_split

from unet_nested import *


torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
        
    """ Define GPU to use """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    """" Input paths """    
    HD = 0; alpha = 1; dist_loss = 0; deep_sup = 0;
    #s_path = './(1) Checkpoints_full_auto_no_spatialW_TRACKER/'; 
    #s_path = './(2) Checkpoints_full_auto_spatialW/'
    
    
    #s_path = './(3) Checkpoints_full_auto_no_spatialW_nested_unet/'
    
    s_path = './(4) Checkpoints_full_auto_no_spatialW_large_TRACKER/'
    
    #s_path = './(5) Checkpoints_full_auto_no_spatialW_large_TRACKER_switch_norm/'
    
    s_path = './(6) Checkpoints_full_auto_no_spatialW_large_TRACKER_NO_NEXT_SEG/'
    
    
    s_path = './(7) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_NO_NEXT_SEG_skipped/'; next_seg = 0;
    
    
    #s_path = './(8) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_YES_NEXT_SEG/'; next_seg = 1;
    
    
    s_path = './(9) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_NO_NEXT_hausdorf/'; next_seg = 0; HD = 1; alpha = 1;
    
    
    #s_path = './(10) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_NO_NEXT/'; next_seg = 0;
    
    
    
    input_path = '/media/user/storage/Data/(2) cell tracking project/a_training_data_GENERATE_FULL_AUTO/Training_cell_track_full_auto_COMPLETED_crop_pads/'

    resume = 0
    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
    onlyfiles_check.sort(key = natsort_key1)
    
    
    # """ load mean and std """  
    mean_arr = np.load('./normalize_pytorch_CLEANED/mean_VERIFIED.npy')
    std_arr = np.load('./normalize_pytorch_CLEANED/std_VERIFIED.npy')       


    num_workers = 2;
    
    
    save_every_num_epochs = 1;    plot_every_num_epochs = 1;    validate_every_num_epochs = 1;  
    
    deep_supervision = False;
    
    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*_crop_input_cur.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,seed          = i.replace('_crop_input_cur.tif','_crop_input_cur_seed.tif'),
                             cur_full_seg  = i.replace('_crop_input_cur.tif','_crop_input_cur_seg_FULL.tif'),
                             next_input    = i.replace('_crop_input_cur.tif','_crop_input_next.tif'),
                             next_full_seg = i.replace('_crop_input_cur.tif','_crop_input_next_seg_FULL.tif'),
                             truth         = i.replace('_crop_input_cur.tif','_crop_truth.tif')) for i in images]
    
    
    
    # ### REMOVE FULL TIMESERIES from training data
    idx_skip = []
    for idx, im in enumerate(examples):
        filename = im['input']
        if 'MOBPF_190105w_1_cuprBZA_10x' in filename:
            print('skip')
            idx_skip.append(idx)
    
    
    ### USE THE EXCLUDED IMAGE AS VALIDATION/TESTING
    examples_test = examples[0:len(idx_skip)]

    examples = [i for j, i in enumerate(examples) if j not in idx_skip]

    
    
    counter = list(range(len(examples)))
    
    
    if not onlyfiles_check:   ### if no old checkpoints found, start new network and tracker
 
        """ Hyper-parameters """
        deep_sup = False
        switch_norm = False
        sp_weight_bool = 0
        #transforms = initialize_transforms(p=0.5)
        #transforms = initialize_transforms_simple(p=0.5)
        transforms = 0
        batch_size = 4;      
        test_size = 0.1  
        
        
        if next_seg:
            in_channels = 4
        else:
            in_channels = 3
        
        

        """ Initialize network """  
        kernel_size = 7
        pad = int((kernel_size - 1)/2)
        unet = UNet_online(in_channels=in_channels, n_classes=2, depth=5, wf=4, kernel_size = kernel_size, padding= int((kernel_size - 1)/2), 
                            batch_norm=True, batch_norm_switchable=switch_norm, up_mode='upsample')
        #unet = NestedUNet(num_classes=2, input_channels=2, deep_sup=deep_sup, padding=pad, batch_norm_switchable=switch_norm)
        #unet = UNet_3Plus(num_classes=2, input_channels=2, kernel_size=kernel_size, padding=pad)

        unet.train()
        unet.to(device)
        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        """ Select loss function *** unimportant if using HD loss """
        if not HD:    loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        else:         loss_function = 'Haussdorf'
            

        """ Select optimizer """
        lr = 1e-5; milestones = [20, 100]  # with AdamW slow down
        optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

        """ Add scheduler """
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
            
        """ initialize index of training set and validation set, split using size of test_size """
        
        idx_train, idx_valid, empty, empty = train_test_split(counter, counter, test_size=test_size, random_state=2018)
        
        """ initialize training_tracker """
        # idx_valid = idx_skip
        # idx_train = counter
        
        tracker = tracker(batch_size, test_size, mean_arr, std_arr, idx_train, idx_valid, deep_sup=deep_sup, switch_norm=switch_norm, alpha=alpha, HD=HD,
                                          sp_weight_bool=sp_weight_bool, transforms=transforms, dataset=input_path)

     
        tracker.next_seg = next_seg
        
    else:             
        """ Find last checkpoint """       
        last_file = onlyfiles_check[-1]
        split = last_file.split('check_')[-1]
        num_check = split.split('.')
        checkpoint = num_check[0]
        checkpoint = 'check_' + checkpoint

        print('restoring weights from: ' + checkpoint)
        check = torch.load(s_path + checkpoint, map_location=lambda storage, loc: storage)
        #check = torch.load(s_path + checkpoint, map_location='cpu')
        #check = torch.load(s_path + checkpoint, map_location=device)
        
        tracker = check['tracker']
        
        unet = check['model_type']
        optimizer = check['optimizer_type']
        scheduler = check['scheduler_type']
        unet.load_state_dict(check['model_state_dict'])
        unet.to(device)
        optimizer.load_state_dict(check['optimizer_state_dict'])
        scheduler.load_state_dict(check['scheduler'])     
        loss_function = check['loss_function']

        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        """ Clean up checkpoint file """
        del check
        torch.cuda.empty_cache()
        
        

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

    """ Create datasets for dataloader """
    training_set = Dataset_tiffs(tracker.idx_train, examples, tracker.mean_arr, tracker.std_arr,
                                           sp_weight_bool=tracker.sp_weight_bool, transforms = tracker.transforms, next_seg=tracker.next_seg)
    val_set = Dataset_tiffs(tracker.idx_valid, examples, tracker.mean_arr, tracker.std_arr,
                                      sp_weight_bool=tracker.sp_weight_bool, transforms = 0, next_seg=tracker.next_seg)
    
    
    """ Create training and validation generators"""
    val_generator = data.DataLoader(val_set, batch_size=tracker.batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True, drop_last = True)

    training_generator = data.DataLoader(training_set, batch_size=tracker.batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=True, drop_last=True)
         
    print('Total # training images per epoch: ' + str(len(training_set)))
    print('Total # validation images: ' + str(len(val_set)))
    

    """ Epoch info """
    train_steps_per_epoch = len(tracker.idx_train)/tracker.batch_size
    validation_size = len(tracker.idx_valid)
    epoch_size = len(tracker.idx_train)    
   

    print('Total # training images per epoch: ' + str(len(training_set)))
    print('Total # validation images: ' + str(len(val_set)))

    """ Epoch info """
    train_steps_per_epoch = len(tracker.idx_train)/tracker.batch_size
    validation_size = len(tracker.idx_valid)
    epoch_size = len(tracker.idx_train)   

    """ Start training """
    starter = 0
    for cur_epoch in range(len(tracker.train_loss_per_epoch), 10000): 
         unet.train        
         loss_train = 0
         jacc_train = 0   
                  
         """ check and plot params during training """             
         for param_group in optimizer.param_groups:
               #tracker.alpha = 0.5
               param_group['lr'] = 1e-6   # manually sets learning rate
               cur_lr = param_group['lr']
               tracker.lr_plot.append(cur_lr)
               tracker.print_essential()

              
              
         iter_cur_epoch = 0;          
         ce_train = 0; dc_train = 0; hd_train = 0;
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
                inputs, labels = transfer_to_GPU(batch_x, batch_y, device,  tracker.mean_arr, tracker.std_arr)
   
                """ zero the parameter gradients"""
                optimizer.zero_grad()       
                
                """ forward + backward + optimize """
                output_train = unet(inputs)

                """ calculate loss: includes HD loss functions """
                if tracker.HD:
                    loss, tracker, ce_train, dc_train, hd_train = compute_HD_loss(output_train, labels, tracker.alpha, tracker, 
                                                                                   ce_train, dc_train, hd_train, val_bool=0)
                else:
                     if deep_sup:   ### IF DEEP SUPERVISION
                        # compute output
                        loss = 0
                        for output in output_train:
                             loss += loss_function(output, labels)
                        loss /= len(output_train)
                        output_train = output_train[-1]  # set this so can eval jaccard later
                   
                     else:   ### IF NORMAL LOSS CALCULATION
                        loss = loss_function(output_train, labels)
                        if torch.is_tensor(spatial_weight):   ### WITH SPATIAL WEIGHTING
                             spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)          
                             weighted = loss * spatial_tensor
                             loss = torch.mean(weighted)
                              
                        else:  ### NO WEIGHTING AT ALL
                             loss = torch.mean(loss)   
                
                
                loss.backward()
                optimizer.step()
               
                """ Training loss """
                """ ********************* figure out how to do spatial weighting??? """
                """ Training loss """
                tracker.train_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                loss_train += loss.cpu().data.numpy()
                
   
                """ Calculate Jaccard on GPU """                 
                jacc = jacc_eval_GPU_torch(output_train, labels)
                jacc = jacc.cpu().data.numpy()
                                            
                jacc_train += jacc # Training jacc
                tracker.train_jacc_per_batch.append(jacc)
   
                tracker.iterations = tracker.iterations + 1       
                iter_cur_epoch += 1
                if tracker.iterations % 100 == 0:
                     print('Trained: %d' %(tracker.iterations))

               
                
    
         tracker.train_loss_per_epoch.append(loss_train/iter_cur_epoch)
         tracker.train_jacc_per_epoch.append(jacc_train/iter_cur_epoch)              
    
         """ Should I keep track of loss on every single sample? and iteration? Just not plot it??? """   
         loss_val = 0; jacc_val = 0
         precision_val = 0; sensitivity_val = 0; val_idx = 0;
         iter_cur_epoch = 0;
         ce_val = 0; dc_val = 0; hd_val = 0;
         total_FPs = 0; total_TPs = 0; total_FNs = 0; total_TNs = 0;
         if cur_epoch % validate_every_num_epochs == 0:
             
              with torch.set_grad_enabled(False):  # saves GPU RAM
                  unet.eval()
                  for batch_x_val, batch_y_val, spatial_weight in val_generator:
                        
                        """ Transfer to GPU to normalize ect... """
                        inputs_val, labels_val = transfer_to_GPU(batch_x_val, batch_y_val, device, mean_arr, std_arr)
                        output_val = unet(inputs_val)
                
                        """ calculate loss 
                                include HD loss functions """
                        if tracker.HD:
                            loss, tracker, ce_val, dc_val, hd_val = compute_HD_loss(output_val, labels_val, tracker.alpha, tracker, 
                                                                                          ce_val, dc_val, hd_val, val_bool=1)
                        else:
                            if deep_sup:                                                
                                # compute output
                                loss = 0
                                for output in output_val:
                                     loss += loss_function(output, labels_val)
                                loss /= len(output_val)                                
                                output_val = output_val[-1]  # set this so can eval jaccard later                            
                            else:
                            
                                loss = loss_function(output_val, labels_val)       
                                if torch.is_tensor(spatial_weight):
                                       spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)          
                                       weighted = loss * spatial_tensor
                                       loss = torch.mean(weighted)
                                elif dist_loss:
                                       loss  # do not do anything if do not need to reduce
                                    
                                else:
                                       loss = torch.mean(loss)  
          
                        """ Training loss """
                        tracker.val_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                        loss_val += loss.cpu().data.numpy()
                                         
                        """ Calculate jaccard on GPU """
                        jacc = jacc_eval_GPU_torch(output_val, labels_val)
                        jacc = jacc.cpu().data.numpy()
                        
                        jacc_val += jacc
                        tracker.val_jacc_per_batch.append(jacc)   


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
                        val_idx = val_idx + tracker.batch_size
                        print('Validation: ' + str(val_idx) + ' of total: ' + str(validation_size))
                        iter_cur_epoch += 1
                 
                
              accuracy = (total_TPs + total_TNs)/(total_TPs + total_TNs + total_FPs + total_FNs)
              sensitivity = total_TPs/(total_TPs + total_FNs);
              precision = total_TPs/(total_TPs + total_FPs);     
                
              tracker.val_loss_per_eval.append(loss_val/iter_cur_epoch)
              tracker.val_jacc_per_eval.append(jacc_val/iter_cur_epoch)   
                   
              tracker.plot_prec.append(precision)
              tracker.plot_sens.append(sensitivity)
              
              tracker.plot_acc.append(accuracy)
                   
                  
              """ Add to scheduler to do LR decay """
              scheduler.step()

         """ calculate new alpha for next epoch """   
         if tracker.HD:
            tracker.alpha = alpha_step(ce_train, dc_train, hd_train, iter_cur_epoch)
                  
         if cur_epoch % plot_every_num_epochs == 0:       
             
            
              """ Plot sens + precision + jaccard + loss """
              plot_metric_fun(tracker.plot_acc, tracker.plot_acc, class_name='', metric_name='accuracy', plot_num=29)
              plt.figure(29); plt.savefig(s_path + 'Accuracy.png')

              plot_metric_fun(tracker.plot_sens, tracker.plot_sens_val, class_name='', metric_name='sensitivity', plot_num=30)
              plt.figure(30); plt.savefig(s_path + 'Sensitivity.png')
                    
              plot_metric_fun(tracker.plot_prec, tracker.plot_prec_val, class_name='', metric_name='precision', plot_num=31)
              plt.figure(31); plt.savefig(s_path + 'Precision.png')
           
              plot_metric_fun(tracker.train_jacc_per_epoch, tracker.val_jacc_per_eval, class_name='', metric_name='jaccard', plot_num=32)
              plt.figure(32); plt.savefig(s_path + 'Jaccard.png')
                   
                 
              plot_metric_fun(tracker.train_loss_per_epoch, tracker.val_loss_per_eval, class_name='', metric_name='loss', plot_num=33)
              plt.figure(33); plt.yscale('log'); plt.savefig(s_path + 'loss_per_epoch.png')          
                   
              
              plot_metric_fun(tracker.lr_plot, [], class_name='', metric_name='learning rate', plot_num=35)
              plt.figure(35); plt.savefig(s_path + 'lr_per_epoch.png') 
     

              """ Plot metrics per batch """                
              plot_metric_fun(tracker.train_jacc_per_batch, [], class_name='', metric_name='jaccard', plot_num=34)
              plt.figure(34); plt.savefig(s_path + 'Jaccard_per_batch.png')
                                
              plot_cost_fun(tracker.train_loss_per_batch, tracker.train_loss_per_batch)                   
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
              
              
              seg_val = np.argmax(output_val[0], axis=-1)  
              
              plot_trainer_3D_PYTORCH_cell_track_AUTO(seg_train, seg_val, batch_x[0, 0:2, ...], batch_x_val[0, 0:2, ...], batch_y[0], batch_y_val[0],
                                       s_path, tracker.iterations, plot_depth=plot_depth)


                                                      
              
         """ To save (every x iterations) """
         if cur_epoch % save_every_num_epochs == 0:                          

               save_name = s_path + 'check_' +  str(tracker.iterations)               
               torch.save({
                'tracker': tracker,

                'model_type': unet,
                'optimizer_type': optimizer,
                'scheduler_type': scheduler,
                
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss_function': loss_function,  
                
                }, save_name)
     

                
               
              
              