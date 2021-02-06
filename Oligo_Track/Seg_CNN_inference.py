# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Tiger

Things to add:
    - GUI!!!
    - need to detect directory change!!!


Seg-CNN:
    Expected input:
        - series of Tiffs
        
        
        - need GPU with at least 6 GB RAM
        
        

"""

import glob, os
""" check if on windows or linux """
if os.name == 'posix':  platform = 'linux'
elif os.name == 'nt': platform = 'windows'
else:
    platform = 0
    

import numpy as np
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
import tkinter
from tkinter import filedialog

import torch
from torch import nn

from UNet_pytorch_online import *
from tracker import *

from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.UNet_functions_PYTORCH import *
import tifffile as tiff

from skimage.filters import threshold_otsu
from skimage.filters import threshold_triangle

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True

""" Define GPU to use """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""  Network Begins: """
s_path = './(19) Checkpoints_TITAN_NO_transforms_AdamW_batch_norm_CLEAN_DATA/'
s_path = './(21) Checkpoints_PYTORCH_NO_transforms_AdamW_batch_norm_CLEAN_DATA_LARGE_NETWORK/'


overlap_percent = 0.5
input_size = 256
depth = 64
num_truth_class = 2

""" TO LOAD OLD CHECKPOINT """
# Read in file names
onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
onlyfiles_check.sort(key = natsort_key1)
      
last_file = onlyfiles_check[-1]
split = last_file.split('check_')[-1]
num_check = split.split('.')
checkpoint = num_check[0]
checkpoint = 'check_' + checkpoint
check = torch.load(s_path + checkpoint, map_location=device)
tracker = check['tracker']

unet = check['model_type']; unet.load_state_dict(check['model_state_dict'])
unet.to(device); unet.eval()
print('parameters:', sum(param.numel() for param in unet.parameters()))


""" Select multiple folders for analysis AND creates new subfolder for results output """
root = tkinter.Tk()
# get input folders
another_folder = 'y';
list_folder = []
input_path = "./"

initial_dir = './'
while(another_folder == 'y'):
    input_path = filedialog.askdirectory(parent=root, initialdir= initial_dir,
                                        title='Please select input directory')
    input_path = input_path + '/'
    
    print('Do you want to select another folder? (y/n)')
    another_folder = input();   # currently hangs forever
    #another_folder = 'y';

    list_folder.append(input_path)
    initial_dir = input_path
        

""" Loop through all the folders and do the analysis!!!"""
for input_path in list_folder:
    foldername = input_path.split('/')[-2]
    sav_dir = input_path + '/' + foldername + '_output_PYTORCH_RETRAINED_105834'

    """ For testing ILASTIK images """
    images = glob.glob(os.path.join(input_path,'*.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('.tif','_truth.tif'), ilastik=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]
     
    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("Directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'
    
    # Required to initialize all
    batch_size = 1;
    
    batch_x = []; batch_y = [];
    weights = [];
    
    plot_jaccard = [];
    
    output_stack = [];
    output_stack_masked = [];
    all_PPV = [];
    input_im_stack = [];
    for i in range(len(examples)):
         
    
        
         """ TRY INFERENCE WITH PATCH-BASED analysis from TORCHIO """
         with torch.set_grad_enabled(False):  # saves GPU RAM            
            input_name = examples[i]['input']  
            input_im = tiff.imread(input_name)

   
            """ Analyze each block with offset in all directions """
            ### CATCH error if too small volume, need to pad with zeros!!!
            if input_im.shape[0] < depth: pad_z = depth 
            else: pad_z = input_im.shape[0] 

            if input_im.shape[1] < input_size: pad_x = input_size 
            else: pad_x = input_im.shape[1] 
            
            if input_im.shape[2] < input_size: pad_y = input_size 
            else: pad_y = input_im.shape[2]             
            
            pad_im = np.zeros([pad_z, pad_x, pad_y])
            pad_im[:input_im.shape[0], :input_im.shape[1], :input_im.shape[2]] = input_im
            input_im = pad_im
            
            
            """ Find reference free SNR """
            all_SNR = [];        
            thresh = threshold_otsu(input_im)
            for slice_depth in range(0, len(input_im) - 33, 33):
            
                first_slices= input_im[slice_depth:slice_depth + 33, ...]
                max_first = plot_max(first_slices, ax=0, plot=0)
                signal = np.mean(np.where(max_first > thresh))
                noise = np.std(np.where(max_first < thresh))
                SNR = 10 * math.log10(signal/noise)
                all_SNR.append(round(SNR, 3))
            all_SNR = np.asarray(all_SNR)
            below_thresh_SNR = np.where(all_SNR < 1.5)[0]
            if len(below_thresh_SNR) > 0:
                print('\nWARNING: SNR is low for image: ' + input_name + 
                      '\n starting at depth slice: ' + str(below_thresh_SNR * 33) + 
                      '\n with SNR values: ' + str(all_SNR[below_thresh_SNR]) )
            
            
        
            """ Start inference on volume """
            print('\nStarting inference on volume: ' + str(i) + ' of total: ' + str(len(examples)))
            segmentation = UNet_inference_by_subparts_PYTORCH(unet, device, input_im, overlap_percent, quad_size=input_size, quad_depth=depth,
                                                      mean_arr=tracker.mean_arr, std_arr=tracker.std_arr, num_truth_class=num_truth_class,
                                                      skip_top=1)
           
            segmentation[segmentation > 0] = 255
            filename = input_name.split('/')[-1].split('.')[0:-1]
            filename = '.'.join(filename)
            

            segmentation = np.asarray(segmentation, np.uint8)
            tiff.imsave(sav_dir + filename + '_' + str(int(i)) +'_segmentation.tif', segmentation)
            segmentation[segmentation > 0] = 1
            
            input_im = np.asarray(input_im, np.uint8)
            tiff.imsave(sav_dir + filename + '_' + str(int(i)) +'_input_im.tif', input_im)
            
        
    print('\n\nSegmented outputs saved in folder: ' + sav_dir)