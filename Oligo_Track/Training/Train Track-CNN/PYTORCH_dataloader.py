# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:10:03 2020

@author: tiger
"""


import torchvision.transforms.functional as TF
import random
from torch.utils import data
import torch
import time
import numpy as np

import scipy
import math


""" Calculate Jaccard on the GPU """
def jacc_eval_GPU_torch(output, truth, ax_labels=-1, argmax_truth=1):
      output = torch.argmax(output,axis=1)
      intersection = torch.sum(torch.sum(output * truth, axis=ax_labels),axis=ax_labels)
      union = torch.sum(torch.sum(torch.add(output, truth)>= 1, axis=ax_labels),axis=ax_labels) + 0.0000001
      jaccard = torch.mean(intersection / union)  # find mean of jaccard over all slices        
      return jaccard

""" Define transforms"""
# import torchio
# from torchio.transforms import (
#     RescaleIntensity,
#     RandomFlip,
#     RandomAffine,
#     RandomElasticDeformation,
#     RandomMotion,
#     RandomBiasField,
#     RandomBlur,
#     RandomNoise,
#     Interpolation,
#     Compose
# )
# from torchio import Image, Subject, ImagesDataset

# def initialize_transforms(p=0.5):
#      transforms = [
#            RandomFlip(axes = 0, flip_probability = 0.5, p = p, seed = None),
           
#            RandomAffine(scales=(0.9, 1.1), degrees=(10), isotropic=False,
#                         default_pad_value='otsu', image_interpolation=Interpolation.LINEAR,
#                         p = p, seed=None),
           
#            # *** SLOWS DOWN DATALOADER ***
#            #RandomElasticDeformation(num_control_points = 7, max_displacement = 7.5,
#            #                         locked_borders = 2, image_interpolation = Interpolation.LINEAR,
#            #                         p = 0.5, seed = None),
#            RandomMotion(degrees = 10, translation = 10, num_transforms = 2, image_interpolation = Interpolation.LINEAR,
#                         p = p, seed = None),
           
#            RandomBiasField(coefficients=0.5, order = 3, p = p, seed = None),
           
#            RandomBlur(std = (0, 4), p = p, seed=None),
           
#            RandomNoise(mean = 0, std = (0, 0.25), p = p, seed = None),
#            #RescaleIntensity((0, 255))
           
#      ]
#      transform = Compose(transforms)
#      return transform


# def initialize_transforms_simple(p=0.5):
#      transforms = [
#            RandomFlip(axes = 0, flip_probability = 0.5, p = p, seed = None),
           
#            #RandomAffine(scales=(0.9, 1.1), degrees=(10), isotropic=False,
#            #             default_pad_value='otsu', image_interpolation=Interpolation.LINEAR,
#            #             p = p, seed=None),
           
#            # *** SLOWS DOWN DATALOADER ***
#            #RandomElasticDeformation(num_control_points = 7, max_displacement = 7.5,
#            #                         locked_borders = 2, image_interpolation = Interpolation.LINEAR,
#            #                         p = 0.5, seed = None),
#            RandomMotion(degrees = 5, translation = 5, num_transforms = 3, image_interpolation = Interpolation.LINEAR,
#                         p = p, seed = None),
           
#            RandomBiasField(coefficients=0.5, order = 3, p = p, seed = None),
           
#            RandomBlur(std = (0, 2), p = p, seed=None),
           
#            RandomNoise(mean = 0, std = (0, 0.25), p = p, seed = None),
#            RescaleIntensity((0, 255))
           
#      ]
#      transform = Compose(transforms)
#      return transform



""" Do pre-processing on GPU
          ***can't do augmentation/transforms here because of CPU requirement for torchio

"""
def transfer_to_GPU(X, Y, device, mean, std, transforms = 0):
     """ Put these at beginning later """
     mean = torch.tensor(mean, dtype = torch.float, device=device, requires_grad=False)
     std = torch.tensor(std, dtype = torch.float, device=device, requires_grad=False)
     
     """ Convert to Tensor """
     inputs = torch.tensor(X, dtype = torch.float, device=device, requires_grad=False)
     labels = torch.tensor(Y, dtype = torch.long, device=device, requires_grad=False)           

     """ Normalization """
     inputs = (inputs - mean)/std
                
     """ Expand dims """
     #inputs = inputs.unsqueeze(1)   

     return inputs, labels


""" Load data directly from tiffs """
import tifffile as tifffile
class Dataset_tiffs(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, examples, mean, std, sp_weight_bool=0, transforms=0, next_seg=1):
        'Initialization'
        #self.labels = labels
        self.list_IDs = list_IDs
        self.examples = examples
        self.transforms = transforms
        self.mean = mean
        self.std = std
        self.sp_weight_bool = sp_weight_bool
        self.next = next_seg

  def apply_transforms(self, image, labels):
        #inputs = np.asarray(image, dtype=np.float32)
        inputs = image

 
        inputs = torch.tensor(inputs, dtype = torch.float,requires_grad=False)
        labels = torch.tensor(labels, dtype = torch.long, requires_grad=False)         
 
        subject_a = Subject(
                one_image=Image(None,  torchio.INTENSITY, inputs),   # *** must be tensors!!!
                a_segmentation=Image(None, torchio.LABEL, labels))
          
        subjects_list = [subject_a]

        subjects_dataset = ImagesDataset(subjects_list, transform=self.transforms)
        subject_sample = subjects_dataset[0]
          
          
        X = subject_sample['one_image']['data'].numpy()
        Y = subject_sample['a_segmentation']['data'].numpy()
        
        return X[0], Y[0]

  def create_spatial_weight_mat(self, labels, edgeFalloff=10,background=0.01,approximate=True):
       
         if approximate:   # does chebyshev
             dist1 = scipy.ndimage.distance_transform_cdt(labels)
             dist2 = scipy.ndimage.distance_transform_cdt(np.where(labels>0,0,1))    # sets everything in the middle of the OBJECT to be 0
                     
         else:   # does euclidean
             dist1 = scipy.ndimage.distance_transform_edt(labels, sampling=[1,1,1])
             dist2 = scipy.ndimage.distance_transform_edt(np.where(labels>0,0,1), sampling=[1,1,1])
             
         """ DO CLASS WEIGHTING instead of spatial weighting WITHIN the object """
         dist1[dist1 > 0] = 0.5
     
         dist = dist1+dist2
         attention = math.e**(1-dist/edgeFalloff) + background   # adds background so no loses go to zero
         attention /= np.average(attention)
         return np.reshape(attention,labels.shape)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)


    
  def append_seed_mask(self, image, seed, cur_full_seg, next_input, next_full_seg) :
    
        """ Append seed to input 
        
                Order of channels is:
                    (1) cur input raw
                    (2) cur input FULL + seed with value == 255
                    (3) next input raw
                    (4) next input FULL
        
        """
        temp = np.zeros((4, ) + np.shape(image))
        temp[0,...] = image
        cur_full_seg[cur_full_seg > 0] = 10   # set these to just be of value 10
        cur_full_seg[seed > 0] = 50
        temp[1,...] = cur_full_seg
        
        temp[2,...] = next_input
        next_full_seg[next_full_seg > 0] = 10  # set these to just be of value 10
        temp[3,...] = next_full_seg
                             
        return temp


  def append_seed_mask_NO_NEXT(self, image, seed, cur_full_seg, next_input) :
    
        """ Append seed to input 
        
                Order of channels is:
                    (1) cur input raw
                    (2) cur input FULL + seed with value == 255
                    (3) next input raw
                    (4) next input FULL
        
        """
        temp = np.zeros((3, ) + np.shape(image))
        temp[0,...] = image
        cur_full_seg[cur_full_seg > 0] = 10   # set these to just be of value 10
        cur_full_seg[seed > 0] = 50
        temp[1,...] = cur_full_seg
        
        temp[2,...] = next_input
        #next_full_seg[next_full_seg > 0] = 10  # set these to just be of value 10
        #temp[3,...] = next_full_seg
                             
        return temp


  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        #X = torch.load('data/' + ID + '.pt')
        #y = self.labels[ID]

 
        input_name = self.examples[ID]['input']
        truth_name = self.examples[ID]['truth']
        
        seed_name = self.examples[ID]['seed']
        cur_full_seg_name = self.examples[ID]['cur_full_seg']
        next_input_name = self.examples[ID]['next_input']
        
        if self.next:
            next_full_seg_name = self.examples[ID]['next_full_seg']
        
        
        """ Read in images """
        X = tifffile.imread(input_name)
        #X = np.expand_dims(X, axis=0)
        Y = tifffile.imread(truth_name)
        Y[Y > 0] = 1
        #Y = np.expand_dims(Y, axis=0)
        
        
        seed = tifffile.imread(seed_name)
        cur_full_seg = tifffile.imread(cur_full_seg_name)
        next_input = tifffile.imread(next_input_name)

        if self.next:
            next_full_seg = tifffile.imread(next_full_seg_name)
        
        
        """ Get spatial weight matrix """
        if self.sp_weight_bool:
             spatial_weight = self.create_spatial_weight_mat(Y)
             
        else:
             spatial_weight = []
             
             
        """ Do normalization here??? """
        #X  = (X  - self.mean)/self.std


        """ Transforms """
        if self.transforms:
              X, Y = self.apply_transforms(X, Y)  
        
        
        
        """ Append seed mask """
        if self.next:
            X = self.append_seed_mask(X, seed, cur_full_seg, next_input, next_full_seg) 
    
        else:        
            X = self.append_seed_mask_NO_NEXT(X, seed, cur_full_seg, next_input) 
        
        
        
        
        # """ If want to do lr_finder """
        # X = np.asarray(X, dtype=np.float32)
        # X = (X - self.mean)/self.std
                    
        # """ Expand dims """
        # #X = inputs.unsqueeze(0)  
        # X = np.expand_dims(X, axis=0)
        # #Y = labels
        # X = torch.tensor(X, dtype = torch.float, requires_grad=False)
        # Y = torch.tensor(Y, dtype = torch.long, requires_grad=False)            
        
        return X, Y, spatial_weight


