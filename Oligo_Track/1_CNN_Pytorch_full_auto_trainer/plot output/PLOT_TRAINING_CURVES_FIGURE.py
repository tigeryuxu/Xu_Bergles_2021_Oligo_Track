#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 19:33:06 2020

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
from skimage import measure
import pandas as pd
import glob, os
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
from tifffile import *
import tkinter
from tkinter import filedialog

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from UNet_pytorch_online import *
from PYTORCH_dataloader import *
from UNet_functions_PYTORCH import *

from matlab_crop_function import *
from plot_functions_CLEANED import *
from data_functions_CLEANED import *
from data_functions_3D import *
from functions_cell_track_auto import *
from skimage.transform import rescale, resize, downscale_local_mean

""" Define transforms"""
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
   Compose
)
from torchio import Image, Subject, ImagesDataset


def my_ceil(a, precision=0):
    return np.round(a + 0.5 * 10**(-precision), precision)

def my_floor(a, precision=0):
    return np.round(a - 0.5 * 10**(-precision), precision)


torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True

""" Define GPU to use """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""  Network Begins: """

s_path = './(10) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_NO_NEXT/'; next_bool = 0; seg_CNN = 0   ### TRACK-CNN
#s_path = '../1_CNN_inference_PYTORCH/(21) Checkpoints_PYTORCH_NO_transforms_AdamW_batch_norm_CLEAN_DATA_LARGE_NETWORK/';  seg_CNN = 1    ### Seg-CNN

""" TO LOAD OLD CHECKPOINT """
# Read in file names
onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
onlyfiles_check.sort(key = natsort_key1)

""" Find last checkpoint """       
last_file = onlyfiles_check[-1]
split = last_file.split('check_')[-1]
num_check = split.split('.')
checkpoint = num_check[0]
checkpoint = 'check_' + checkpoint
num_check = int(num_check[0])

check = torch.load(s_path + checkpoint, map_location=device)

unet = check['model_type']; unet.load_state_dict(check['model_state_dict'])
unet.to(device); unet.eval()
#unet.training # check if mode set correctly

print('parameters:', sum(param.numel() for param in unet.parameters()))



# """ load mean and std """  
input_path = './normalize_pytorch_CLEANED/'
mean_arr = np.load(input_path + 'mean_VERIFIED.npy')
std_arr = np.load(input_path + 'std_VERIFIED.npy')

"""
    Re-plot stuff from tracker
"""

""" Set globally """
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rcParams['figure.dpi'] = 300
ax_title_size = 20
leg_size = 18


tracker = check['tracker']
#plot_tracker(tracker, sav_dir)

sav_dir = s_path
# plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
# plt.rcParams['axes.xmargin'] = 0.5
# plt.rcParams['axes.ymargin'] = 0
    
plot_metric_fun(tracker.train_jacc_per_epoch, tracker.val_jacc_per_eval, class_name='', metric_name='jaccard', plot_num=32,
                location='lower right', leg_size=leg_size)
plt.xlabel('Epochs', fontsize=ax_title_size); plt.ylabel('Jaccard', fontsize=ax_title_size)
#plt.yticks(np.arange(0.12, 0.35, 0.04))
plt.xticks(np.arange(0, len(tracker.train_jacc_per_epoch)+1, 5.0))
ax = plt.gca()
rs = ax.spines["right"]; rs.set_visible(False)
ts = ax.spines["top"]; ts.set_visible(False)
plt.tight_layout(); 

plt.legend(frameon=False, fontsize=leg_size, loc='lower right')
top = my_ceil(np.max(tracker.train_jacc_per_epoch), precision=1)
bottom = my_floor(np.min(tracker.train_jacc_per_epoch), precision=1)
plt.ylim([bottom, top])

plt.savefig(sav_dir + 'Jaccard.png')

   
plot_metric_fun(tracker.train_loss_per_epoch, tracker.val_loss_per_eval, class_name='', metric_name='loss', plot_num=33,
                location='upper right', leg_size=leg_size)

plt.xlabel('Epochs', fontsize=ax_title_size); plt.ylabel('Loss', fontsize=ax_title_size)
#plt.yticks(np.arange(0.12, 0.35, 0.04))
plt.xticks(np.arange(0, len(tracker.train_jacc_per_epoch)+1, 5.0))
ax = plt.gca()
rs = ax.spines["right"]; rs.set_visible(False)
ts = ax.spines["top"]; ts.set_visible(False)
plt.tight_layout(); 

plt.legend(frameon=False, fontsize=leg_size,loc='upper right')
top = my_ceil(np.max(tracker.train_loss_per_epoch), precision=2)
bottom = my_floor(np.min(tracker.train_loss_per_epoch), precision=2)


plt.figure(33); plt.yscale('log'); plt.tight_layout();
plt.ylim([10 ** -4, 10 ** 0])
plt.savefig(sav_dir + 'loss_per_epoch.png')          
                

if not seg_CNN:
    plot_metric_fun([], tracker.plot_acc, class_name='', metric_name='accuracy', plot_num=29,
                    location='lower right', leg_size=leg_size)
    plt.xlabel('Epochs', fontsize=ax_title_size); plt.ylabel('Accuracy', fontsize=ax_title_size)
    #plt.yticks(np.arange(0.12, 0.35, 0.04))
    plt.xticks(np.arange(0, len(tracker.train_jacc_per_epoch)+1, 5.0))       
    ax = plt.gca()
    rs = ax.spines["right"]; rs.set_visible(False)
    ts = ax.spines["top"]; ts.set_visible(False)
    
    
    plt.legend(frameon=False, fontsize=leg_size,loc='lower right')
    
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('orange')
    plt.ylim([0, 1])
    plt.figure(29); plt.tight_layout(); plt.savefig(sav_dir + 'Accuracy.png')
    print('Final accuracy: ' + str(tracker.plot_acc[-1]))
    
    
plot_metric_fun(tracker.plot_sens_val, tracker.plot_sens, class_name='', metric_name='sensitivity', plot_num=30,
                location='lower right', leg_size=leg_size)
plt.xlabel('Epochs', fontsize=ax_title_size); plt.ylabel('Sensitivity', fontsize=ax_title_size)
plt.yticks(np.arange(0.8, 1.05, 0.05))        
plt.xticks(np.arange(0, len(tracker.train_jacc_per_epoch)+1, 5.0))
ax = plt.gca()
rs = ax.spines["right"]; rs.set_visible(False)
ts = ax.spines["top"]; ts.set_visible(False)

plt.legend(frameon=False, fontsize=leg_size,loc='lower right')
leg = ax.get_legend()
leg.legendHandles[0].set_color('orange')
#plt.ylim([0, 1])
plt.figure(30); plt.tight_layout(); plt.savefig(sav_dir + 'Sensitivity.png')
print('Final sensitivity: ' + str(tracker.plot_sens[-1]))

      
plot_metric_fun(tracker.plot_prec_val, tracker.plot_prec, class_name='', metric_name='precision', plot_num=31,
                location='lower right', leg_size=leg_size)
plt.xlabel('Epochs', fontsize=ax_title_size); plt.ylabel('Precision', fontsize=ax_title_size)
#plt.yticks(np.arange(0.12, 0.35, 0.04))
plt.xticks(np.arange(0, len(tracker.train_jacc_per_epoch)+1, 5.0))               
ax = plt.gca()
rs = ax.spines["right"]; rs.set_visible(False)
ts = ax.spines["top"]; ts.set_visible(False)

plt.legend(frameon=False, fontsize=leg_size,loc='lower right')
leg = ax.get_legend()
leg.legendHandles[0].set_color('orange')
plt.ylim([0, 1])
plt.figure(31); plt.tight_layout(); plt.savefig(sav_dir + 'Precision.png')
print('Final precision: ' + str(tracker.plot_prec[-1]))


print('Num training samples: ' + str(len(tracker.idx_train)))
print('Num validation samples: ' + str(len(tracker.idx_valid)))













