# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================
@author: Tiger

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
from UNet_functions_PYTORCH import *
from PYTORCH_dataloader import *
from HD_loss import *

from sklearn.model_selection import train_test_split

from unet_nested import *


import torchio
from torchio.transforms import (
   RescaleIntensity,
   RandomFlip,
   RandomAffine,
   RandomElasticDeformation,
   RandomMotion,
   RandomBiasField,
   RandomBlur,
   RandomNoise,
   Interpolation,
   Resample,
   Compose
)
from torchio import Image, Subject, ImagesDataset
           

from functions_transforms import *


torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
        
    """ Define GPU to use """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    """" Input paths """    
    HD = 0; alpha = 1; dist_loss = 0; deep_sup = 0; 
    #s_path = './(8) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_YES_NEXT_SEG/'; next_seg = 1;
    #s_path = './(9) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_NO_NEXT_hausdorf/'; next_seg = 0; HD = 1; alpha = 1;
    s_path = './(10) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_NO_NEXT/'; next_seg = 0;



    sav_dir = s_path + 'test_transforms'
    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("Directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'    

    
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

    """ USE VALIDATION DATA FOR THIS """
    examples = examples_test

    #examples = [i for j, i in enumerate(examples) if j not in idx_skip]
    counter = list(range(len(examples)))
    
               
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
    

    """ Create datasets for dataloader """
    training_set = Dataset_tiffs(counter, examples, tracker.mean_arr, tracker.std_arr,
                                           sp_weight_bool=tracker.sp_weight_bool, transforms = tracker.transforms, next_seg=tracker.next_seg)

    """ Create training and validation generators"""
    training_generator = data.DataLoader(training_set, batch_size=1, shuffle=False, num_workers=num_workers,
                      pin_memory=True, drop_last=True)
        
    """ Start training """
    starter = 0
    unet.eval()
    iterator = 0
    for batch_x, batch_y, spatial_weight in training_generator:
                    
        # PRINT OUT THE SHAPE OF THE INPUT
 
        """ Plot for debug """    
        np_inputs = np.asarray(batch_x.numpy()[0], dtype=np.uint8)
        m = plot_max(np_inputs[0], ax=0, plot=0)
        #plot_max(np_inputs[1], ax=0, plot = 0)
        #plot_max(np_inputs[2], ax=0)
        
        crop_im = np_inputs[0]
        crop_cur_seg = np_inputs[1]
        crop_next_input = np_inputs[2]
        m_next = plot_max(crop_next_input, ax=0, plot=0)
        
        
        np_labels = np.asarray(batch_y.numpy()[0], dtype=np.uint8)
        np_labels[np_labels > 0] = 255
        m_lab = plot_max(np_labels, ax=0, plot=0)
        
        """ save to find image you want """
        imsave(sav_dir + str(iterator) + '_input.tif', np.uint8(m))
        imsave(sav_dir + str(iterator) + '_NEXT_input.tif', np.uint8(m_next))
        imsave(sav_dir + str(iterator) + '_TRUTH.tif', np.uint8(m_lab))
        
        if iterator == 71:
            break;
        
        if iterator != 71:
            
            iterator += 1
            continue;
            
        
            
        p = 1 
        next_bool = 0
     
        ### decide which image to run the transform on
        transform_next = 0
        
        target_shape = [crop_im.shape[1], crop_im.shape[2], crop_im.shape[0]]
        """ Transforms to try:
                (1) blur
                (2) motion blur
                (3) bias field
                (4) random noise
                
                (5) affine deformation
                (6) elastic deformation
                
                
                (7) RESAMPLE/down and upsample???
             
        """      

        """ Run through diff iterations of transforms """
        
        
        all_jacc = []
        for val in range(2, 8, 2):
            transforms = define_transform(transform='blur', p=1, blur_std=val, motion_trans=10, biascoeff=0.5, noise_std=0.25, affine_deg=10, elastic_disp=7.5)

            jacc, X = test_transform(unet, transforms, crop_im, np_labels, crop_cur_seg, crop_next_input, 
                                  mean_arr, std_arr, device, sav_dir, transform_type='1_blur', val=val)
            all_jacc.append(jacc)
            plot_max(X)

        all_jacc = []
        

        for val in range(4, 10, 2):
            transforms = define_transform(transform='motion', p=1, blur_std=4,
                                            motion_deg=val,motion_trans=val, motion_num=val, biascoeff=0.5, noise_std=0.25, affine_deg=10, elastic_disp=7.5)

            jacc, X = test_transform(unet, transforms, crop_im, np_labels, crop_cur_seg, crop_next_input, 
                                  mean_arr, std_arr, device, sav_dir, transform_type='2_motion', val=val, transform_next=transform_next)
            all_jacc.append(jacc)
            
            plot_max(X)
            
            
        all_jacc = []
        for val in np.arange(0, 2, 0.5):
            transforms = define_transform(transform='biasfield', p=1,
                                          biascoeff=val)

            jacc, X = test_transform(unet, transforms, crop_im, np_labels, crop_cur_seg, crop_next_input, 
                                  mean_arr, std_arr, device, sav_dir, transform_type='3_biasfield', val=val, transform_next=transform_next)
            all_jacc.append(jacc)
            plot_max(X)
            
            
            
        all_jacc = []
        for val in np.arange(1, 50, 10):
            transforms = define_transform(transform='noise', p=1, blur_std=4, motion_trans=10, biascoeff=0.5, 
                                          noise_std=val, affine_deg=10, elastic_disp=7.5)

            jacc, X = test_transform(unet, transforms, crop_im, np_labels, crop_cur_seg, crop_next_input, 
                                  mean_arr, std_arr, device, sav_dir, transform_type='4_noise', val=val, transform_next=transform_next)
            all_jacc.append(jacc)
            plot_max(X)
            
            
            
        # all_jacc = []
        # for val in range(4, 10, 2):
        #     transforms = define_transform(transform='affine', p=1, blur_std=4, motion_trans=10, biascoeff=0.5, noise_std=0.25,
        #                                   affine_deg=val, affine_trans=val, elastic_disp=7.5)

        #     jacc, X = test_transform(unet, transforms, crop_im, np_labels, crop_cur_seg, crop_next_input, 
        #                           mean_arr, std_arr, device, sav_dir, transform_type='5_affine', val=val, transform_next=transform_next)
        #     all_jacc.append(jacc)
        #     plot_max(X)
            
            
        all_jacc = []
        for val in range(8, 16, 2):
            transforms = define_transform(transform='elastic', p=1, 
                                          elastic_disp=val)

            jacc, X = test_transform(unet, transforms, crop_im, np_labels, crop_cur_seg, crop_next_input, 
                                  mean_arr, std_arr, device, sav_dir, transform_type='6_elastic', val=val, transform_next=transform_next, resample=1)
            all_jacc.append(jacc)     
            plot_max(X)
            

        all_jacc = []
        for val in np.arange(0.5, 1.5, 0.25):
            transforms = define_transform(transform='resample', p=1, blur_std=4, motion_trans=10, biascoeff=0.5, noise_std=0.25, affine_deg=10, 
                                          elastic_disp=7.5, resample_size=val, target_shape=crop_im.shape)

            jacc, X = test_transform(unet, transforms, crop_im, np_labels, crop_cur_seg, crop_next_input, 
                                  mean_arr, std_arr, device, sav_dir, transform_type='7_resample', val=val, transform_next=0, resample=1)
            all_jacc.append(jacc)    
            plot_max(X)
    
    
              