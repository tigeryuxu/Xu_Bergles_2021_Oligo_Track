# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Tiger
"""

#import tensorflow as tf
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
from skimage import measure
import scipy
import cv2 as cv
from natsort import natsort_keygen, ns

from plot_functions_CLEANED import *
from data_functions_CLEANED import *
from data_functions_3D import *
#from UNet import *
#from UNet_3D import *
import glob, os
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

#from csbdeep.internals import predict
from tifffile import *
import tkinter
from tkinter import filedialog

import pandas as pd
from skimage import measure


""" Required to allow correct GPU usage ==> or else crashes """
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.keras.backend.set_session(tf.Session(config=config))


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

#from UNet_pytorch import *
from UNet_pytorch_online import *
from PYTORCH_dataloader import *
from UNet_functions_PYTORCH import *
from matlab_crop_function import *

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True

""" Define GPU to use """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""  Network Begins: """
s_path = './(1) Checkpoints_full_auto_no_spatialW/'

#s_path = './(2) Checkpoints_full_auto_spatialW/'

crop_size = 160
z_size = 32
num_truth_class = 2

lowest_z_depth = 135;
lowest_z_depth = 150;

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

unet = check['model_type']
unet.load_state_dict(check['model_state_dict'])
unet.to(device)
unet.eval()
#unet.training # check if mode set correctly

print('parameters:', sum(param.numel() for param in unet.parameters()))

# """ load mean and std """  
input_path = './normalize_pytorch_CLEANED/'
mean_arr = np.load(input_path + 'mean_VERIFIED.npy')
std_arr = np.load(input_path + 'std_VERIFIED.npy')


""" Select multiple folders for analysis AND creates new subfolder for results output """
root = tkinter.Tk()
# get input folders
another_folder = 'y';
list_folder = []
input_path = "./"

initial_dir = '/media/user/storage/Data/'
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
    sav_dir = input_path + '/' + foldername + '_output_FULL_AUTO'

    """ For testing ILASTIK images """
    #images = glob.glob(os.path.join(input_path,'*_single_channel.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    #images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    #examples = [dict(input=i,seg=i.replace('_single_channel.tif','_single_channel_segmentation.tif'), truth=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]


    images = glob.glob(os.path.join(input_path,'*_input_im.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,seg=i.replace('_input_im.tif','_segmentation.tif'), truth=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]




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
    
    

   
    """ Initialize matrix of cells """
    num_timeseries = len(examples)
    columns = list(range(0, num_timeseries))
    matrix_timeseries = pd.DataFrame(columns = columns)
    
   
    input_name = examples[0]['input']            
    input_im = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
    depth_total, empty, empty = input_im.shape
    
    input_im = input_im[0:lowest_z_depth, ...]
    input_im = np.moveaxis(input_im, 0, -1)
 
    seg_name = examples[0]['seg']  
    cur_seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
    cur_seg = cur_seg[0:lowest_z_depth, ...]
    cur_seg = np.moveaxis(cur_seg, 0, -1)
     
    #height_tmp, width_tmp, depth_tmp = input_im.shape
    """ loop through each cell in cur_seg and find match in next_seg
     
          ***keep track of double matched cells
          ***append to dataframe
    """
    cur_seg[cur_seg > 0] = 1
    labelled = measure.label(cur_seg)
    cur_cc = measure.regionprops(labelled)
    
    
    tracked_cells_df = pd.DataFrame(columns = {'SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z', 'coords', 'visited'})     
    
    
    ### add the cells from the first frame into "tracked_cells" matrix
    for cell in cur_cc:
         if not np.isnan(np.max(tracked_cells_df.SERIES)):
              series = np.max(tracked_cells_df.SERIES) + 1
         else:
                   series = 1
         centroid = cell['centroid']
  
         """ SKIP IF HAVE TO MOVE THE CROPPING BOX IN THE BOTTOM Z-dimension """
         if int(centroid[2]) + z_size/2 >= lowest_z_depth:
               continue
          
        
         coords = cell['coords']
         row = {'SERIES': series, 'COLOR': 'BLANK', 'FRAME': 0, 'X': int(centroid[0]), 'Y':int(centroid[1]), 'Z': int(centroid[2]), 'coords':coords, 'visited': 0}
         tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)
          
    
    
    ### HACK: ??? #####################################################################################################################################
    """ WHICH WAY IS CORRECT??? """
    width_tmp, height_tmp, depth_tmp = input_im.shape
    
    """ Get truth from .csv as well """
    truth = 1
    scale = 1
    
    if truth:
         #truth_name = 'MOBPF_190627w_5_syglassCorrectedTracks.csv'     #### full auto from the past: truth_name = 'MOBPF_190626w_4_10x_output_PYTORCH_output_FULLY_AUTO.csv'
                                                                          ### TP: 3412, FP: 7, FN: 41, TN: 4 ==> total mistakes 115 + 145/135 out of total 1379 cells
                                                                               
                                                                          
                                                                          ### with list excluision 63 bad ==> of 1307 
                                                                          
                                                                          
                                                                          
                                                                          ### with Spatial weight:
                                                                              # TPs == 3386, FPs == 7, TNs == 3, FNs == 21, mistakes == 62  (or 126 + 27/129)
                                                                              # ==> total 1375 cells
                                                                          
                                                                          
                                                                           
                                                                          
         
         
         truth_name = 'MOBPF_190626w_4_syGlassEdited_20200607.csv'   # cuprizone
         

                   ### total 1387 , TP == 1926, FP == 64, FN == 173, TN == 1010
                        ### mistracked == 130  and 162 or 157 + 56
                        
                        
                   ### with spatial Weighting, TP == 1956, FP == 96, FN == 117, TN == 982
                       ### mistracked 130  out of 1394
                   
                   
                        
         
         

         #truth_name = 'a1901128-r670_syGlass_20x.csv'
         
         
         #truth_name = '680_syGlass_10x.csv'    #### TP: 2084, TN: 440, FN: 45, FP: 21, all cells == 711, mistakes 119 (~65 is by 1)  + 51 negatives (over-tracked - doubles???)
                                                 ### no clearing of singles at end + no clearing of doubly counted
                                                 
                                                 
                                                 ### 135 mistakes > 0 if use np.unique at the end (i.e. ignoring doubles)
                                                 
                                                 
                                                 
                                              ### SPATIAL WEIGHT 
                                              ### TP: 2076, TN: 414, FP: 37, FN: 32
                                                  # 52 mistakes of 637
                                              
                                                  # or 48 + 5/   or 124 + 5/129
                                              
                                              
                                              
                                                 
                                                 
                                                 
                                                 
         
         
         truth_cur_im, truth_array  = gen_truth_from_csv(frame_num=0, input_path=input_path, filename=truth_name, 
                            input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
         
         
         truth_output_df = pd.DataFrame(columns = {'SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z'})
         
         
         """ Add the very first timeframe of cells """
         # for cell_idx in np.where(truth_array.FRAME == 0)[0]:
         #      x = truth_array.X[cell_idx]
         #      y = truth_array.Y[cell_idx]
         #      z = truth_array.Z[cell_idx]
              
         #      if cur_seg[int(x), int(y), int(z)] > 0:
         #                      row  = truth_array.iloc[cell_idx]
         #                      truth_output_df = truth_output_df.append(row)                                 
         
     
    TN = 0; TP = 0; FN = 0; FP = 0; doubles = 0; extras = 0; skipped = 0; blobs = 0; not_registered = 0; double_linked = 0; seg_error = 0;
    
    list_exclude = [];
    
    for i in range(1, len(examples)):
         print('Starting inference on volume: ' + str(i) + ' of total: ' + str(len(examples)))
       
         with torch.set_grad_enabled(False):  # saves GPU RAM            

            """ Gets next seg as well """
            input_name = examples[i]['input']            
            next_input = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
            next_input = next_input[0:lowest_z_depth, ...]
            next_input = np.moveaxis(next_input, 0, -1)
     
   
            seg_name = examples[i]['seg']  
            next_seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
            next_seg = next_seg[0:lowest_z_depth, ...]
            next_seg = np.moveaxis(next_seg, 0, -1)
            
            
            """ Get truth for next seg as well """
            if truth:
                 truth_next_im, truth_array  = gen_truth_from_csv(frame_num=i, input_path=input_path, filename=truth_name, 
                              input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
                     

            iterator = 0;
            
            for cell_idx in np.where(tracked_cells_df.visited == 0)[0]: 
                 
                 cell = tracked_cells_df.iloc[cell_idx]
                 
                 ### go to unvisited cells
                 x = cell.X; y = cell.Y; z = cell.Z;
                 coords = cell.coords                 
                 series = cell.SERIES
                 
                 ### SO DON'T VISIT AGAIN
                 tracked_cells_df.visited[cell_idx] = 1
                 
                 
                 """ DONT TEST IF TOO SMALL """
                 if len(coords) < 10:
                      continue;
                 
                 
                 blank_im = np.zeros(np.shape(input_im))
                 blank_im[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
                 
                 
                 crop_cur_seg, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(cur_seg, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
                 
                 """ SKIP IF HAVE TO MOVE THE CROPPING BOX IN THE BOTTOM Z-dimension """
                 if z + z_size/2 >= lowest_z_depth:
                      print('skip')
                      skipped += 1
                      continue
                 
                 """ TRY REGISTRATION??? """
                 # import SimpleITK as sitk
                 # elastixImageFilter = sitk.ElastixImageFilter()
                 # elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(crop_im))
                 # elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(crop_next_input))
                 # elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
                 # im = elastixImageFilter.Execute()
                 # im_arr = sitk.GetArrayFromImage(im)
                 # im_arr[im_arr >= 255] = 255
                 # im_arr[im_arr < 0] = 0
                 # #sitk.WriteImage(elastixImageFilter.GetResultImage())
                 # crop_next_input = im_arr;

                 crop_seed, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(blank_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
                 crop_im, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(input_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
                 crop_cur_seg[crop_cur_seg > 0] = 10
                 crop_cur_seg[crop_seed > 0] = 50                 
                    
                 crop_next_input, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(next_input, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                         
                 crop_next_seg, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(next_seg, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
                 crop_next_seg[crop_next_seg > 0] = 10
               
                 
                 """ Get ready for inference """
                 batch_x = np.zeros((4, ) + np.shape(crop_im))
                 batch_x[0,...] = crop_im
                 batch_x[1,...] = crop_cur_seg
                 batch_x[2,...] = crop_next_input
                 batch_x[3,...] = crop_next_seg
                 batch_x = np.moveaxis(batch_x, -1, 1)
                 batch_x = np.expand_dims(batch_x, axis=0)
                 
                 ### NORMALIZE
                 batch_x = normalize(batch_x, mean_arr, std_arr)
                 
                 
                 """ Convert to Tensor """
                 inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)

                 # forward pass to check validation
                 output_val = unet(inputs_val)

                 """ Convert back to cpu """                                      
                 output_val = output_val.cpu().data.numpy()            
                 output_val = np.moveaxis(output_val, 1, -1)
                 seg_train = np.argmax(output_val[0], axis=-1)  
                 seg_train = np.moveaxis(seg_train, 0, -1)
                 
                 iterator += 1


                 """ ***IF MORE THAN ONE OBJECT IS IN FINAL SEGMENTATION, choose the best matched one!!!"""
                 label = measure.label(seg_train)
                 cc_seg_train = measure.regionprops(label)
                 if len(cc_seg_train) > 1:
                      doubles += 1
                      print('multi objects in seg')
                      
                      
                      ### pick the ideal one ==> confidence??? distance??? and then set track color to 'YELLOW'
                      """ Best one is one that takes up most of the area in the "next_seg" """
                      #add = crop_next_seg + seg_train
                      label = measure.label(crop_next_seg)
                      cc_next = measure.regionprops(label)
                      
                      
                      best = np.zeros(np.shape(crop_next_seg))
                      
                      all_ratios = []
                      for multi_check in cc_seg_train:
                           coords_m = multi_check['coords']
                           crop_next_seg[coords_m[:, 0], coords_m[:, 1], coords_m[:, 2]]
                           #print('out')
                           for seg_check in cc_next:
                                coords_n = seg_check['coords']
                                if np.any((coords_m[:, None] == coords_n).all(-1).any(-1)):   ### overlapped
                                     ratio = len(coords_m)/len(coords_n)
                                     all_ratios.append(ratio)
                                     #print('in')
                         
                            
                      if len(all_ratios) > 0:
                          best_coords = cc_seg_train[all_ratios.index(max(all_ratios))]['coords']
                          best[best_coords[:, 0], best_coords[:, 1], best_coords[:, 2]] = 1
                      seg_train = best
                      label = measure.label(seg_train)
                      cc_seg_train = measure.regionprops(label)                      

                 
                 """ Find coords of identified cell and scale back up, later find which ones in next_seg have NOT been already identified
                 """
                 if len(cc_seg_train) > 0:
                      next_coords = cc_seg_train[0].coords
                      next_coords = scale_coords_of_crop_to_full(next_coords, box_x_min , box_y_min, box_z_min)
                      
                      next_centroid = np.asarray(cc_seg_train[0].centroid)
                      next_centroid[0] = np.int(next_centroid[0] + box_x_min)   # SCALING the ep_center
                      next_centroid[1] = np.int(next_centroid[1] + box_y_min)
                      next_centroid[2] = np.int(next_centroid[2] + box_z_min)
                     
                      #next_seg[int(next_centroid[0]), int(next_centroid[1]), int(next_centroid[2])]                 
                      
                      ### add to matrix 
                      row = {'SERIES': series, 'COLOR': 'GREEN', 'FRAME': i, 'X': int(next_centroid[0]), 'Y':int(next_centroid[1]), 'Z': int(next_centroid[2]), 'coords':next_coords, 'visited': 0}
                      tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)     


                         
                      """ FIND DOUBLES EARLY TO CORRECT AS YOU GO """
                      if np.any(next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] == 250): ### if this place has already been visited in the past
                           print('double_linked')
                           double_linked += 1
                           
                           
                           ### pick the ideal one ==> confidence??? distance??? and then set track color to 'YELLOW'
                           
                           """ Find series number that matches in index == next_frame """
                           dup_series = []
                           for idx_next in np.where(tracked_cells_df.FRAME == i)[0]:
                                
                                if tracked_cells_df.X[idx_next] == next_centroid[0] and tracked_cells_df.Y[idx_next] == next_centroid[1] and tracked_cells_df.Z[idx_next] == next_centroid[2]:
                                     dup_series.append(tracked_cells_df.SERIES[idx_next])
                                     
                                     
                           """ Get location of cell on previous frame corresponding to these SERIES numbers
                                     and find cell that is CLOSEST
                           
                           """
                           all_dist = []
                           for dup in dup_series:
                                index = np.where((tracked_cells_df["SERIES"] == dup) & (tracked_cells_df["FRAME"] == i - 1))[0]
                                x_check = tracked_cells_df.X[index];
                                y_check = tracked_cells_df.Y[index];
                                z_check = tracked_cells_df.Z[index];
                                
                                sub = np.copy(next_centroid)
                                sub[0] = (sub[0] - x_check) * 0.083
                                sub[1] = (sub[1] - y_check) * 0.083
                                sub[2] = (sub[2] - z_check) * 3
                                
                                dist = np.linalg.norm(sub)
                                all_dist.append(dist)
                                
                           closest = all_dist.index(min(all_dist))
                            
                           """ drop everything else that is not close and set their series to be RED, set the closest to be YELLO """
                           keep_series = dup_series[closest]
                           tracked_cells_df.COLOR[tracked_cells_df["SERIES"] == keep_series] = 'YELLOW'
                            
                            
                           dup_series = np.delete(dup_series, np.where(np.asarray(dup_series) == keep_series)[0])
                           for dup in dup_series:
                                 tracked_cells_df.COLOR[tracked_cells_df["SERIES"] == dup] = 'RED'
                                 
                                 ### also delete the 2nd occurence of it
                                 tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[np.where(((tracked_cells_df["SERIES"] == dup) & (tracked_cells_df["FRAME"] == i)))[0]])

                      
                      next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 250;   ### set current one to be value 2 so in future will know has already been identified
                      
       
                 

                 """ Check if TP, TN, FP, FN """
                 if truth:
                      crop_truth_cur, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(truth_cur_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                         
                      crop_truth_next, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid_with_pads(truth_next_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
     
                      """ DEBUG """
                        # plot_max(crop_cur_seg, ax=-1)
                        # plot_max(crop_im, ax=-1)
                        # plot_max(crop_next_input, ax=-1)
                        # plot_max(crop_next_seg, ax=-1)
                        # plot_max(seg_train, ax=-1) 
                        # plot_max(crop_truth_cur, ax=-1)
                        # plot_max(crop_truth_next, ax=-1)
                        # plt.pause(0.0005)
                        # plt.pause(0.0005)                      
                      print('TPs = ' + str(TP) + '; FPs = ' + str(FP) + '; TNs = ' + str(TN) + '; FNs = ' + str(FN) + '; extras = ' + str(extras))
                      
                      seg_train = dilate_by_ball_to_binary(seg_train, radius = 3)  ### DILATE A BIT
                      crop_next_seg = dilate_by_ball_to_binary(crop_next_seg, radius = 3)  ### DILATE A BIT
                      crop_seed = dilate_by_ball_to_binary(crop_seed, radius = 3)
                      
                      
                      """ REMOVE EVERYTHING IN CROP_NEXT_SEG THAT DOES NOT MATCH WITH SOMETHING CODY PUT UP, to prevent FPs  of unknown checking"""
                      # if nothing in the second frame
                      value_cur_frame = np.unique(crop_truth_cur[crop_seed > 0])
                      value_cur_frame = np.delete(value_cur_frame, np.where(value_cur_frame == 0)[0][0])  # DELETE zero
                      
                      values_next_frame = np.unique(crop_truth_next[crop_next_seg > 0])
                      
                      ### skip if no match on cur frame in truth
                      if len(value_cur_frame) == 0:
                           not_registered += 1;
                           print('not_registered')
                           continue
                      
                      
                      
                      if not np.any(np.in1d(value_cur_frame, values_next_frame)):   ### if it does NOT exist on next frame               
                           ### BUT IF EXISTS IN GENERAL ON 2nd frame, just not in the segmentation, then skip ==> is segmentation missed error
                           values_next_frame_all = np.unique(crop_truth_next[crop_truth_next > 0])
                           if np.any(np.in1d(value_cur_frame, values_next_frame_all)):
                                seg_error += 1;
                                print('seg_error')
                                list_exclude.append(value_cur_frame[0])
                                continue; # SKIP
                           
                           ### count blobs:
                           if len(value_cur_frame) > 1:
                                blobs += 1
                         
                           ### AND if seg_train says it does NOT exist ==> then is TRUE NEGATIVE
                           if np.count_nonzero(seg_train) == 0:       
                                TN += 1     
                           else:
                                ### otherwise, all of the objects are False positives
                                
                                #FP += len(cc_seg_train)
                                FP += 1
                                
                      else:
                           
                           ### (1) if seg_train is empty ==> then is a FALSE NEGATIVE
                           if np.count_nonzero(seg_train) == 0:
                                #print('depth_cur: ' + str(np.where(crop_truth_cur == np.max(value_cur_frame))[-1][0])) 
                                #print('depth_next: ' + str(np.where(crop_truth_next == np.max(value_cur_frame))[-1][0]))
                                FN += 1
                                
                                """ Missing a lot of little tiny ones that are dim
                                          *** maybe at the end add these back in??? by just finding nearest unassociated ones???
                                
                                """
                           else:
                                
                              ### (2) find out if seg_train has identified point with same index as previous frame
                              values_next_frame = np.unique(crop_truth_next[seg_train > 0])
                                                          
                              values_next_frame = np.delete(values_next_frame, np.where(values_next_frame == 0)[0][0])  # delete zeros
                           
                                
                              if np.any(np.in1d(value_cur_frame, values_next_frame)):
                                     TP += 1
                                     values_next_frame = np.delete(values_next_frame, np.where(values_next_frame == values_next_frame)[0][0])
                                
                                
                              
                              """ if this is first time here, then also add the ones from initial index """
                              if i == 1:
                                   row  = truth_array[(truth_array["SERIES"] == np.max(value_cur_frame)) & (truth_array["FRAME"] == 0)]
                                   truth_output_df = truth_output_df.append(row)                                      
                              row  = truth_array[(truth_array["SERIES"] == np.max(value_cur_frame)) & (truth_array["FRAME"] == i)]
                              truth_output_df = truth_output_df.append(row) 
                              
                              
                                
                              # but if have more false positives
                              if len(values_next_frame) > 0:
                                #FP += len(values_next_frame)
                                extras += len(values_next_frame)
                 
                           
                 
                      plt.close('all')  
                      
                                    
                 print('Testing cell: ' + str(iterator) + ' of total: ' + str(len(np.where(tracked_cells_df.visited == 0)[0]))) 



            """ associate remaining cells that are "new" cells and add them to list to check as well as the TRUTH tracker """
            bw_next_seg = np.copy(next_seg)
            bw_next_seg[bw_next_seg > 0] = 1
            
            labelled = measure.label(bw_next_seg)
            next_cc = measure.regionprops(labelled)
              
              
            ### add the cells from the first frame into "tracked_cells" matrix
            num_new = 0; num_new_truth = 0
            for cell in next_cc:
               coords = cell['coords']
               
               
               if not np.any(next_seg[coords[:, 0], coords[:, 1], coords[:, 2]] == 250):   ### 250 means already has been visited
                    #print(np.unique(truth_next_im[coords[:, 1], coords[:, 1], coords[:, 2]]))
                    #print(coords)
                    series = np.max(tracked_cells_df.SERIES) + 1
     
                    centroid = cell['centroid']
             
                    """ SKIP IF HAVE TO MOVE THE CROPPING BOX IN THE BOTTOM Z-dimension """
                    if int(centroid[2]) + z_size/2 >= lowest_z_depth:
                          continue
                     
                    
                    row = {'SERIES': series, 'COLOR': 'BLANK', 'FRAME': i, 'X': int(centroid[0]), 'Y':int(centroid[1]), 'Z': int(centroid[2]), 'coords':coords, 'visited': 0}
                    tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)
                    
                    
                    
                    """ Add to TRUTH as well """
                    value_next_frame = np.max(truth_next_im[coords[:, 0], coords[:, 1], coords[:, 2]])
                    if value_next_frame > 0:
                         row  = truth_array[(truth_array["SERIES"] == np.max(value_next_frame)) & (truth_array["FRAME"] == i)]
                         truth_output_df = truth_output_df.append(row) 
                         print('value_next_frame')
                         num_new_truth += 1
                         
                    num_new += 1
                         
                    
            
            """ Set next frame to be current frame """
            input_im = next_input
            cur_seg = next_seg
            cur_seg[cur_seg > 0] = 0
            truth_cur_im = truth_next_im
            
            
            
            
    """ Parse the old array: """
    print('duplicates: ' + str(np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME']))))    ### *** REAL DUPLICATES
    tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]]
    
    print('double_linked throughout analysis: ' + str(double_linked))
    #print('num_blobs: ' + str(num_blobs))
    
    
    #print('duplicates: ' + str(np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'SERIES']))))   ### cell in same location across frames
    #tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'SERIES'], keep=False))[0]]

    
    ### (1) unsure that all of 'RED' or 'YELLOW' are indicated as such
    ### ***should be fine, just turn all "BLANK" into "GREEN"  
    tracked_cells_df.COLOR[tracked_cells_df['COLOR'] == 'BLANK'] = 'GREEN'
    
    num_YELLOW = 0; num_RED = 0 
    for cell_num in np.unique(tracked_cells_df.SERIES):
       
        color_arr = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].COLOR)
        
        if np.any(color_arr == 'RED'):
            tracked_cells_df.COLOR[np.where(tracked_cells_df.SERIES == cell_num)[0]] = 'RED'
            num_RED += 1
            
        elif np.any(color_arr == 'YELLOW'):
            tracked_cells_df.COLOR[np.where(tracked_cells_df.SERIES == cell_num)[0]] = 'YELLOW'
            num_YELLOW += 1
        
    
    
    """ Pre-save everything """
    tracked_cells_df = tracked_cells_df.sort_values(by=['SERIES', 'FRAME'])
    tracked_cells_df.to_csv(sav_dir + 'tracked_cells_df_RAW.csv', index=False)
    #tracked_cells_df = pd.read_csv(sav_dir + 'tracked_cells_df_RAW.csv', sep=',')
    
    
    ### (2) remove everything only on a single frame, except for very first frame
    singles = []
    for cell_num in np.unique(tracked_cells_df.SERIES):
          
               track_length_SEG = len(np.unique(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME))         


               """ remove anything that's only tracked for length of 1 timeframe """
               """ excluding if that timeframe is the very first one OR the very last one"""
               
               #print(truth_output_df.FRAME[truth_output_df.SERIES == cell_num] )
               if len(np.where(np.asarray(track_length_SEG) == 1)[0]) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == 0) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == np.max(tracked_cells_df.FRAME)):
                   singles.append(cell_num)
                   tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[np.where(tracked_cells_df.SERIES == cell_num)])
                   continue;
                        

   

    
    if truth:
         truth_array.to_csv(sav_dir + 'truth_array.csv', index=False)
         truth_output_df = truth_output_df.sort_values(by=['SERIES'])
         truth_output_df.to_csv(sav_dir + 'truth_output_df.csv', index=False)
         truth_output_df = pd.read_csv(sav_dir + 'truth_output_df.csv', sep=',')

    
    
    """  Save images in output """
    input_name = examples[0]['input']
    filename = input_name.split('/')[-1]
    filename = filename.split('.')[0:-1]
    filename = '.'.join(filename)
    
    for frame_num in range(len(examples)):
         
         output_frame = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im)
         im = convert_matrix_to_multipage_tiff(output_frame)
         imsave(sav_dir + filename + '_' + str(frame_num) + '_output.tif', im)
         
         
         output_frame = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, color=1)
         im = convert_matrix_to_multipage_tiff(output_frame)
         imsave(sav_dir + filename + '_' + str(frame_num) + '_output_COLOR.tif', im)


         """ Also save image with different colors for RED/YELLOW and GREEN"""
         
         # input_name = examples[frame_num]['input']            
         # next_input = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
         # next_input = next_input[0:lowest_z_depth, ...]
         # next_input = np.moveaxis(next_input, 0, -1)
      
         # if truth:
         #      seg_truth_compare = gen_im_frame_from_TRUTH_array(truth_output_df, frame_num, input_im, lowest_z_depth, height_tmp, width_tmp, depth_tmp, scale=0)
         #      truth_im = gen_im_frame_from_TRUTH_array(truth_array, frame_num, input_im, lowest_z_depth, height_tmp, width_tmp, depth_tmp, scale=0)
              
              
              

                
     ### (3) drop other columns
    tracked_cells_df = tracked_cells_df.drop(columns=['visited', 'coords'])
    
    
    ### and reorder columns
    cols =  ['SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z']
    tracked_cells_df = tracked_cells_df[cols]

    ### (4) save cleaned
                        
    tracked_cells_df.to_csv(sav_dir + 'tracked_cells_df_clean.csv', index=False)               
            
            
            
            
            
            
    """ Compare dataframes to see of the tracked cells, how well they were tracked """    
    itera = 0
    if truth:
         all_lengths = []
         truth_lengths = []
         output_lengths = []
         for cell_num in np.unique(truth_output_df.SERIES):
               
              
               
               ### EXCLUDE SEG_ERRORS
               if not np.any( np.in1d(list_exclude, cell_num)):
               #if not np.any( np.in1d(list_exclude, cell_num)) and np.any( np.in1d(all_cell_nums, cell_num)):
                   #track_length_SEG =  len(np.where(truth_output_df.SERIES == cell_num)[0])
                    #track_length_TRUTH = len(np.where(truth_array.SERIES == cell_num)[0])
     
     
                    track_length_TRUTH  = len(np.unique(truth_array.iloc[np.where(truth_array.SERIES == cell_num)].FRAME))
                    track_length_SEG = len(np.unique(truth_output_df.iloc[np.where(truth_output_df.SERIES == cell_num)].FRAME))         
     
     
                    """ remove anything that's only tracked for length of 1 timeframe """
                    """ excluding if that timeframe is the very first one """
                    
                    #print(truth_output_df.FRAME[truth_output_df.SERIES == cell_num] )
                    if len(np.where(np.asarray(track_length_SEG) == 1)[0]) and not np.any(truth_output_df.iloc[np.where(truth_output_df.SERIES == cell_num)].FRAME == 0) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == np.max(tracked_cells_df.FRAME)):
                        continue;

     
                    all_lengths.append(track_length_TRUTH - track_length_SEG)
                    
                    truth_lengths.append(track_length_TRUTH)
                    output_lengths.append(track_length_SEG)
                    
                    
                    if track_length_TRUTH - track_length_SEG > 0 or track_length_TRUTH - track_length_SEG < 0:
                         #print(truth_array.FRAME[truth_array.SERIES == cell_num])
                         #print("truth is: " + str(np.asarray(truth_array.iloc[np.where(truth_array.SERIES == cell_num)].FRAME)))
                         #print("output is: " + str(np.asarray(truth_output_df.iloc[np.where(truth_output_df.SERIES == cell_num)].FRAME)))
                         
                         itera += 1
                         
                         #if len(np.asarray(truth_output_df.iloc[np.where(truth_output_df.SERIES == cell_num)].FRAME)) == 0:
                         #           zzz
                    
                    

                    
    plt.figure(); plt.plot(all_lengths)
    print(len(all_lengths))
    print(len(np.where(np.asarray(all_lengths) > 0)[0]))
    print(len(np.where(np.asarray(all_lengths) < 0)[0]))
    #truth_output_df = truth_output_df.sort_values(by=['SERIES'])

    
    fig, ax = plt.subplots()
    y_pos = np.arange(len(all_lengths))
    ax.barh(y_pos, truth_lengths)
    ax.barh(y_pos, output_lengths)
    
    
    

    """ Load old .csv and plot it??? """
    #MATLAB_name = 'MOBPF_190627w_5_output_FULL_AUTO.csv'
    #MATLAB_name = 'MOBPF_190626w_4_10x_output_PYTORCH_output_FULLY_AUTO.csv'
    MATLAB_name = 'output.csv'
    
    
    MATLAB_auto_array = pd.read_csv(input_path + MATLAB_name, sep=',')
    
    
    all_cells_MATLAB = np.unique(MATLAB_auto_array.SERIES)
    all_cells_TRUTH = np.unique(truth_array.SERIES)


    all_lengths = []
    truth_lengths = []    
    MATLAB_lengths = []
    
    
    all_cell_nums = []
    for i in range(len(examples)):
         print('Starting inference on volume: ' + str(i) + ' of total: ' + str(len(examples)))
       
         seg_name = examples[i]['seg']  
         seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
         seg = seg[0:lowest_z_depth, ...]
         seg = np.moveaxis(seg, 0, -1)       
         
         
    
         truth_next_im, truth_array  = gen_truth_from_csv(frame_num=i, input_path=input_path, filename=truth_name, 
                                   input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
                          
         
         MATLAB_next_im, MATLAB_auto_array  = gen_truth_from_csv(frame_num=i, input_path=input_path, filename=MATLAB_name, 
                                   input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp,
                                   depth_tmp=depth_total, scale=0, swap=1)
         
         
         """ region props on seg ==> then loop through each individual cell to find out which numbers are matched
         
              then delete those numbers from the "all_cells_MATLAB" and "all_cells_TRUTH" matrices
              
              while finding out the lengths
         """
         label_seg = measure.label(seg)
         cc_seg = measure.regionprops(label_seg)
         
         for cell in cc_seg:
              coords = cell['coords']
              
              
              if np.any(truth_next_im[coords[:, 0], coords[:, 1], coords[:, 2]] > 0) and np.any(MATLAB_next_im[coords[:, 0], coords[:, 1], coords[:, 2]] > 0):
                   
                   num_truth = np.unique(truth_next_im[coords[:, 0], coords[:, 1], coords[:, 2]])
                   if len(np.intersect1d(all_cells_TRUTH, num_truth)) > 0:
                        num_new_truth = np.max(np.intersect1d(all_cells_TRUTH, num_truth)) ### only keep cells that haven't been tracked before                   
                        track_length_TRUTH = len(truth_array[truth_array.SERIES == num_new_truth])
                        all_cells_TRUTH = all_cells_TRUTH[all_cells_TRUTH != num_new_truth]   # remove this cell so cant be retracked
                        
                        
                   """ get cell from MATLAB """
                   num_MATLAB = np.unique(MATLAB_next_im[coords[:, 0], coords[:, 1], coords[:, 2]])   
                   if len(np.intersect1d(all_cells_MATLAB, num_MATLAB)) > 0:
                        num_new_MATLAB = np.max(np.intersect1d(all_cells_MATLAB, num_MATLAB)) ### only keep cells that haven't been tracked before
                        track_length_MATLAB = len(truth_array[MATLAB_auto_array.SERIES == num_new_MATLAB])
                        all_cells_MATLAB = all_cells_MATLAB[all_cells_MATLAB != num_new_MATLAB]   # remove this cell so cant be retracked
                        
                   
                   if len(np.intersect1d(all_cells_TRUTH, num_truth)) > 0 and len(np.intersect1d(all_cells_MATLAB, num_MATLAB)) > 0:
                   
                        all_lengths.append(track_length_TRUTH - track_length_MATLAB)
                        truth_lengths.append(track_length_TRUTH)
                        MATLAB_lengths.append(track_length_MATLAB)   
                        
                        all_cell_nums.append(num_new_truth)
                   
                   
    plt.figure(); plt.plot(all_lengths)
    print(len(np.where(np.asarray(all_lengths) > 0)[0]))
    print(len(np.where(np.asarray(all_lengths) < 0)[0]))         
                                   
    
    
    """ Parse the old array: """
    print('duplicates: ' + str(np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'FRAME']))))    ### *** REAL DUPLICATES
    MATLAB_auto_array.iloc[np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]]
    
    #print('duplicates: ' + str(np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'SERIES']))))   ### cell in same location across frames
    #MATLAB_auto_array.iloc[np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'SERIES'], keep=False))[0]]
    
               #duplicates: 378  
    
     
    
     
    
     
    """ **** SEE ONLY THE CELLS THAT WERE TRACKED IN MATLAB, TRUTH, and CNN!!! """
     
    
     
     
    
    """ Things to fix still:
         
         (1) doubles assign
         
         (2) ***blobs
         
         
         does using del [] on double tracked cells do anything bad???
    
         
    
        ***FINAL OUTPUT:
                - want to show on tracked graph:
                        (a) cells tracked over time, organize by mistakes highest at top horizontal bar graph ==> also only used cells matched across all 3 matrices
                        (b) show number of cells tracked by each method
                        (c) show # of double_linked to be resolved
                        
    
    
    """
    
    
    
    
    