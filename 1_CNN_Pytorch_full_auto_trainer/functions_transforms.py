#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:48:12 2020

@author: user
"""
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
   CropOrPad,
   Compose
)
from torchio import Image, Subject, ImagesDataset
            
import torch

""" Transforms to try:
        (1) blur
        (2) motion blur
        (3) bias field
        (4) random noise
        
        (5) affine deformation
        (6) elastic deformation
        
        (7) Resample/downsample
     
"""        
def define_transform(transform, p, blur_std=4, motion_trans=10, motion_deg=10, motion_num=2, biascoeff=0.5, noise_std=0.25, affine_trans=10, affine_deg=10, elastic_disp=7.5, resample_size=1, target_shape=0):
    ### (1) try with different blur
    if transform == 'blur':
        transforms = [RandomBlur(std = (blur_std, blur_std), p = p, seed=None)]; transforms = Compose(transforms)
   
    ### (2) try with different motion artifacts
    if transform == 'motion':
        transforms = [RandomMotion(degrees = motion_deg, translation = motion_trans, num_transforms = motion_num, image_interpolation = Interpolation.LINEAR,
                        p = p, seed = None),]; transforms = Compose(transforms)
    ### (3) with random bias fields
    if transform == 'biasfield':
        transforms = [RandomBiasField(coefficients = biascoeff, order = 3, p = p, seed = None)]; transforms = Compose(transforms)

    ### (4) try with different noise artifacts
    if transform == 'noise':
        transforms = [RandomNoise(mean = 0, std = (noise_std, noise_std), p = p, seed = None)]; transforms = Compose(transforms)

   
    ### (5) try with different warp (affine transformatins)
    if transform == 'affine':
        transforms = [RandomAffine(scales=(1, 1), degrees=(affine_deg), isotropic=False,
                            default_pad_value='otsu', image_interpolation=Interpolation.LINEAR,
                            p = p, seed=None)]; transforms = Compose(transforms)                    

    ### (6) try with different warp (elastic transformations)
    if transform == 'elastic':
        transforms = [RandomElasticDeformation(num_control_points = elastic_disp, max_displacement = 20,
                                        locked_borders = 2, image_interpolation = Interpolation.LINEAR,
                                        p = p, seed = None),]; transforms = Compose(transforms)
        
        
    if transform == 'resample':
        transforms = [Resample(target = resample_size, image_interpolation = Interpolation.LINEAR,
                                        p = p),
                      
                      
                      CropOrPad(target_shape = target_shape, p=1)
                      ]; 
        
        
        
        transforms = Compose(transforms)
        
        
    return transforms



""" Run an image with transform """

def test_transform(unet, transforms, crop_im, np_labels, crop_cur_seg, crop_next_input, mean_arr, std_arr, device, sav_dir, transform_type, val, transform_next=0, resample=0):
    ### transforms to apply to crop_im 


    if not transform_next:
        inputs = crop_im
        
        
    else:
        inputs = crop_next_input
            
    inputs = torch.tensor(inputs, dtype = torch.float,requires_grad=False)
    
    crop_cur_seg_tensor = torch.tensor(crop_cur_seg, dtype = torch.float,requires_grad=False)
    crop_next_tensor = torch.tensor(crop_next_input, dtype = torch.float,requires_grad=False)
    crop_im_tensor = torch.tensor(crop_im, dtype = torch.float,requires_grad=False)
    
    #labels = torch.tensor(labels, dtype = torch.long, requires_grad=False)         
    labels = np_labels
    labels = torch.tensor(labels, dtype = torch.float,requires_grad=False)
   
    subject_a = Subject(
            one_image=Image(None,  torchio.INTENSITY, inputs),   # *** must be tensors!!!
            a_segmentation=Image(None, torchio.LABEL, labels),
            a_cur_seg=Image(None, torchio.LABEL, crop_cur_seg_tensor),
            a_next_im=Image(None, torchio.INTENSITY, crop_next_tensor),
            a_cur_im=Image(None, torchio.INTENSITY, crop_im_tensor),
            
            
            )
     
    subjects_list = [subject_a]
   
    subjects_dataset = ImagesDataset(subjects_list, transform=transforms)
    subject_sample = subjects_dataset[0]
     
     
    """ MUST ALSO TRANSFORM THE SEED IF IS ELASTIC, rotational transformation!!!"""
     
    X = subject_sample['one_image']['data'].numpy()
    Y = subject_sample['a_segmentation']['data'].numpy()
    
    
    if resample:
        np_labels = Y[0]
        
        
        crop_cur_seg = subject_sample['a_cur_seg']['data'].numpy()[0]
        if not transform_next:
            crop_next_input = subject_sample['a_next_im']['data'].numpy()[0]
        else:
            crop_im = subject_sample['a_cur_im']['data'].numpy()[0]
        
    
    # if next_bool:
    #     batch_x = np.zeros((4, ) + np.shape(crop_im))
    #     batch_x[0,...] = X
    #     batch_x[1,...] = crop_cur_seg
    #     batch_x[2,...] = crop_next_input
    #     batch_x[3,...] = crop_next_seg
    #     batch_x = np.moveaxis(batch_x, -1, 1)
    #     batch_x = np.expand_dims(batch_x, axis=0)
   
    # else:
    if not transform_next:
        batch_x = np.zeros((3, ) + np.shape(crop_im))
        batch_x[0,...] = X
        batch_x[1,...] = crop_cur_seg
        batch_x[2,...] = crop_next_input
        #batch_x[3,...] = crop_next_seg
        #batch_x = np.moveaxis(batch_x, -1, 1)
        batch_x = np.expand_dims(batch_x, axis=0)
        
    else:
        batch_x = np.zeros((3, ) + np.shape(crop_im))
        batch_x[0,...] = crop_im
        batch_x[1,...] = crop_cur_seg
        batch_x[2,...] = X
        #batch_x[3,...] = crop_next_seg
        #batch_x = np.moveaxis(batch_x, -1, 1)
        batch_x = np.expand_dims(batch_x, axis=0)        
        
       
   
    ### NORMALIZE
    batch_x = normalize(batch_x, mean_arr, std_arr)                 

    ### Convert to Tensor
    inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)
    
    np_labels = np.expand_dims(np_labels, axis=0)
    np_labels[np_labels > 0] = 1
    labels = torch.tensor(np_labels, dtype = torch.float, device=device, requires_grad=False)
   
    # forward pass to check validation
    output_train = unet(inputs_val)
   
    """ Convert back to cpu """                                      
    output_val = np.moveaxis(output_train.cpu().data.numpy(), 1, -1)      
    seg_train = np.moveaxis(np.argmax(output_val[0], axis=-1), 0, -1)
   
    
   
    normalized = (X-np.min(X))/(np.max(X)-np.min(X))
   
    
    m_in = plot_max(normalized[0], ax=0, plot=0)
    m_cur_s = plot_max(crop_cur_seg, ax=0, plot=0)
    m_next = plot_max(crop_next_input, ax=0, plot=0)
    m_labels = plot_max(np_labels[0], ax=0, plot=0)
    m_OUT = plot_max(seg_train, ax=-1, plot=0)
    
    
    imsave(sav_dir + transform_type + '_val_' + str(val) + '_1_input.tif', np.uint8(m_in * 255))
    m_cur_s[m_cur_s == 50] = 255
    m_cur_s[m_cur_s == 10] = 50
    imsave(sav_dir + transform_type + '_val_' + str(val) + '_2_cur_seg.tif', np.uint8(m_cur_s))
    imsave(sav_dir + transform_type + '_val_' + str(val) + '_3_next_input.tif', np.uint8(m_next))
    imsave(sav_dir + transform_type + '_val_' + str(val) + '_4_labels.tif', np.uint8(m_labels) * 255)
    imsave(sav_dir + transform_type + '_val_' + str(val) + '_5_OUTPUT.tif', np.uint8(m_OUT) * 255)
        
    """ Training loss """
    """ ********************* figure out how to do spatial weighting??? """
    """ Training loss """
    #tracker.train_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
    #loss_train += loss.cpu().data.numpy()
    
    """ Calculate Jaccard on GPU """                
    
    jacc = jacc_eval_GPU_torch(output_train, labels)
    jacc = jacc.cpu().data.numpy()
    print(jacc)
                                 
    return jacc, X[0]
