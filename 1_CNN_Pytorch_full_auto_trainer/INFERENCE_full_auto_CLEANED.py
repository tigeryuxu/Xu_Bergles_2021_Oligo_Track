# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Tiger


To run:
    
    (1) press run
    (2) select folder where the OUTPUT of the segmentation-CNN in the previous step is located
    
    (3) change the "lowest_z_depth" variable as needed (to save computational time)
            - i.e. if you want to go down to 100 slices, then select 100 + 2 == 120 for lowest_z_depth
                b/c the last 20 z-slices are discarded to account for possible large shifts in tissue movement
                so always segment 20 slices more than you actually care about
    
    



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

from PLOT_FIGURES_functions import *


import pandas as pd
import scipy.stats as sp
import seaborn as sns

""" optional dataviewer if you want to load it """
# import napari
# with napari.gui_qt():
#     viewer = napari.view_image(seg_val)


""" Define transforms"""
    

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True


""" Set globally """
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)


""" Define GPU to use """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""  Network Begins: """

s_path = './(7) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_NO_NEXT_SEG_skipped/'; next_bool = 0;
#s_path = './(8) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_YES_NEXT_SEG/'; next_bool = 1;


s_path = './(10) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_NO_NEXT/'; next_bool = 0;


lowest_z_depth = 180;

crop_size = 160
z_size = 32
num_truth_class = 2
min_size = 10
both = 0
elim_size = 100

exclude_side_px = 40

min_size = 100;
upper_thresh = 800;

scale_xy = 0.83
scale_z = 3

control = 0;

""" AMOUNT OF EDGE TO ELIMINATE 


    scaling???
"""
scale_for_animation = 0


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

net_two_identified = 0;
net_two_tested = 0;
    
for input_path in list_folder:
    foldername = input_path.split('/')[-2]
    sav_dir = input_path + '/' + foldername + '_output_FULL_AUTO_no_next_10_125762_TEST_7'


    """ For testing ILASTIK images """
    # images = glob.glob(os.path.join(input_path,'*_single_channel.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    # images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    # examples = [dict(input=i,seg=i.replace('_single_channel.tif','_single_channel_segmentation.tif'), truth=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]

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
    
    
    

    """ Initialize matrix of cells """   
    input_name = examples[0]['input']            
    input_im = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
    depth_total, empty, empty = input_im.shape
    
    #input_im = input_im[0:lowest_z_depth, ...]
    input_im = np.moveaxis(input_im, 0, -1)
    width_tmp, height_tmp, depth_tmp = input_im.shape
    
    if scale_for_animation:
        copy_input_im = np.copy(input_im)
    
    
    
    seg_name = examples[0]['seg']  
    cur_seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
    #cur_seg = cur_seg[0:lowest_z_depth, ...]
    cur_seg[lowest_z_depth:-1, ...] = 0   ### CLEAR EVERY CELL BELOW THIS DEPTH
    
    cur_seg = np.moveaxis(cur_seg, 0, -1)
     
    """ loop through each cell in cur_seg and find match in next_seg
    """
    cur_seg[cur_seg > 0] = 1
    labelled = measure.label(cur_seg)
    cur_cc = measure.regionprops(labelled)
    tracked_cells_df = pd.DataFrame(columns = {'SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z', 'coords', 'visited'})     
    
    
    """ add the cells from the first frame into "tracked_cells" matrix """ 
    for cell in cur_cc:
         if not np.isnan(np.max(tracked_cells_df.SERIES)):
              series = np.max(tracked_cells_df.SERIES) + 1
         else:
                   series = 1
         centroid = cell['centroid']
  
         """ SKIP IF HAVE TO MOVE THE CROPPING BOX IN THE BOTTOM Z-dimension """
         # if int(centroid[2]) + z_size/2 >= lowest_z_depth:
         #       continue
         
         coords = cell['coords']
         
         
         """ DONT TEST IF TOO SMALL """
         if len(coords) < min_size:
              continue;         
         
         # if series == 39:
         #     zzz
         row = {'SERIES': series, 'COLOR': 'BLANK', 'FRAME': 0, 'X': int(centroid[0]), 'Y':int(centroid[1]), 'Z': int(centroid[2]), 'coords':coords, 'visited': 0}
         tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)
          
    
    """ Get truth from .csv as well """
    truth = 0
    scale = 1
    
    if truth:
         #truth_name = 'MOBPF_190627w_5_syglassCorrectedTracks.csv'; scale = 0
         #truth_name = 'MOBPF_190626w_4_syGlassEdited_20200607.csv';  scale = 1  # cuprizone
         #truth_name = 'a1901128-r670_syGlass_20x.csv';    # gets hazy at the end
         truth_name = '680_syGlass_10x.csv'                  

         #truth_name = 'MOBPF_190106w_5_cuprBZA_10x.tif - T=0_650_syGlass_10x.csv'   # well registered and clean window except for single frame

         
         
         truth_cur_im, truth_array  = gen_truth_from_csv(frame_num=0, input_path=input_path, filename=truth_name, 
                            input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
         truth_output_df = pd.DataFrame(columns = {'SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z'})
         
    else:
        truth_cur_im = 0; truth_array = 0; truth_output_df = 0;
         
    
    
    """ Start looping through segmented nuclei """
    list_exclude = [];
    TN = 0; TP = 0; FN = 0; FP = 0; doubles = 0; extras = 0; skipped = 0; blobs = 0; not_registered = 0; double_linked = 0; seg_error = 0;
    
    animator_iterator = 0;
    all_size_pairs = []
    for frame_num in range(1, len(examples)):
         print('Starting inference on volume: ' + str(frame_num) + ' of total: ' + str(len(examples)))
         all_dup_indices = [];
        
         with torch.set_grad_enabled(False):  # saves GPU RAM            

            """ Gets next seg as well """
            input_name = examples[frame_num]['input']            
            next_input = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
            #next_input = np.moveaxis(next_input[0:lowest_z_depth, ...], 0, -1)
            next_input = np.moveaxis(next_input, 0, -1)
            if scale_for_animation:
                copy_next_input = np.copy(next_input)
            
            
            seg_name = examples[frame_num]['seg']  
            next_seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
            next_seg[lowest_z_depth:-1, ...] = 0   ### CLEAR EVERY CELL BELOW THIS DEPTH
            next_seg = np.moveaxis(next_seg, 0, -1)


            """ Plot for animation """
            if scale_for_animation:
                track_cur_seg = np.zeros(np.shape(next_seg))
                track_new_seg = np.zeros(np.shape(next_seg))
                track_term_seg = np.zeros(np.shape(next_seg))
                plot_next = np.zeros(np.shape(next_seg))
                
            """ Get truth for next seg as well """
            if truth:
                 truth_next_im, truth_array  = gen_truth_from_csv(frame_num=frame_num, input_path=input_path, filename=truth_name, 
                              input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
           
            """ Iterate through all cells """
            iterator = 0;            
            
            cell_size = []
            size_pairs = []
            for cell_idx in progressbar.progressbar(np.where(tracked_cells_df.visited == 0)[0], max_value=len(np.where(tracked_cells_df.visited == 0)[0]), redirect_stdout=True): 
                 
                 cell = tracked_cells_df.iloc[cell_idx]
                 
                 ### go to unvisited cells
                 x = cell.X; y = cell.Y; z = cell.Z;

                 ### SO DON'T VISIT AGAIN
                 tracked_cells_df.visited[cell_idx] = 1
                 
                 
                 """ DONT TEST IF TOO SMALL """
                 cell_size.append(len(cell.coords))
                 if len(cell.coords) < 20:
                           continue;
                 
                 """ Crop and prep data for CNN, ONLY for the first 2 frames so far"""
                 batch_x, crop_im, crop_cur_seg, crop_seed, crop_next_input, crop_next_seg, crop_next_seg_non_bin, box_xyz, box_over = prep_input_for_CNN(cell, input_im, next_input, cur_seg,
                                                                                                         next_seg, mean_arr, std_arr, x, y, z, crop_size, z_size,
                                                                                                         height_tmp, width_tmp, depth_tmp, next_bool=next_bool)

                 ### Convert to Tensor
                 inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)

                 # forward pass to check validation
                 output_val = unet(inputs_val)

                 """ Convert back to cpu """                                      
                 output_val = np.moveaxis(output_val.cpu().data.numpy(), 1, -1)      
                 seg_train = np.moveaxis(np.argmax(output_val[0], axis=-1), 0, -1)

                 iterator += 1


                 """ ***IF MORE THAN ONE OBJECT IS IN FINAL SEGMENTATION, choose the best matched one!!!

                         *** NEED TO FIX THIS TOO!!!
                 
                 """
                 cc_seg_train, seg_train = select_one_from_excess(seg_train, crop_next_seg)
                 
                      
                 """ if the next segmentation is EMPTY OR if it's tiny, just skip it b/c might be error noise output """
                 if len(cc_seg_train) == 0 or len(cc_seg_train[0].coords) < 20:                      
                      debug = 0                      
                 else:
                                          
                     """ Check if current volume is much smaller than ENORMOUS next volume """
                     size_pairs.append([len(cell.coords), len(cc_seg_train[0].coords)])
                     #if len(cell.coords) < 500 and len(cc_seg_train[0].coords) > 1000:
                     if (len(cell.coords) > 750 or len(cc_seg_train[0].coords) > 1500) and len(cc_seg_train[0].coords) > len(cell.coords) * 2:
                              
                            cc_seg_train = []
                            #print('yo, initial smaller than next by a lot, dont use')
        
       
                     """ Find coords of identified cell and scale back up, later find which ones in next_seg have NOT been already identified
                     """
                     new = 0
                     if len(cc_seg_train) > 0:
                          next_coords = cc_seg_train[0].coords                            
    
                          next_coords = scale_coords_of_crop_to_full(next_coords, box_xyz, box_over)
                          
                          next_centroid = np.asarray(cc_seg_train[0].centroid)
                          next_centroid = scale_single_coord_to_full(next_centroid, box_xyz, box_over)
                          
                          
                          """ Need to check to ensure the coords do not go over the image size limit because the CNN output is NOT subjected to the limits!!! """
                          next_coords = check_limits([next_coords], width_tmp, height_tmp, depth_tmp)[0]
                          next_centroid = check_limits_single([next_centroid], width_tmp, height_tmp, depth_tmp)[0]
    



                          """ FIND DOUBLES EARLY TO CORRECT AS YOU GO """
                          if np.any(next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] == 250): ### if this place has already been visited in the past
                               #print('double_linked'); 
                               double_linked += 1                    
                               
                               tmp_check_dup = tracked_cells_df.copy()
                               ### add to matrix 
                               row = {'SERIES': cell.SERIES, 'COLOR': 'GREEN', 'FRAME': frame_num, 'X': int(next_centroid[0]), 'Y':int(next_centroid[1]), 'Z': int(next_centroid[2]), 'coords':next_coords, 'visited': 0}
                               tmp_check_dup = tmp_check_dup.append(row, ignore_index=True)  
                                              
                               #tmp_check_dup, dup_series = sort_double_linked(tmp_check_dup, next_centroid, frame_num)     
                               
                               dup_series = tmp_check_dup.iloc[np.where(tmp_check_dup.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]].SERIES
                               dup_series = np.asarray(dup_series)
                               #print('find dupliocates')        
                               
                               all_dup_indices = np.concatenate((all_dup_indices, dup_series))                               
                            
                          else:
                              ### add to matrix 
                              row = {'SERIES': cell.SERIES, 'COLOR': 'GREEN', 'FRAME': frame_num, 'X': int(next_centroid[0]), 'Y':int(next_centroid[1]), 'Z': int(next_centroid[2]), 'coords':next_coords, 'visited': 0}
                              tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)     

                              """ set current one to be value 2 so in future will know has already been identified """
                              next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 250;  
                              new = 1;
                          
                     else:                           
                          ####   DEBUG if not matched
                          len(np.where(crop_seed)[0])         
                          
                            
                     """ Check if TP, TN, FP, FN """
                     if False:
                          TP, FP, TN, FN, extras, blobs, truth_output_df, truth_array, list_exclude = parse_truth(truth_cur_im,  truth_array, truth_output_df, truth_next_im, 
                                                                                                                  seg_train, crop_next_seg, crop_seed, list_exclude, frame_num, x, y, z, crop_size, z_size,
                                                                                                                  blobs, TP, FP, TN, FN, extras, height_tmp, width_tmp, depth_tmp)
                     """ Plot for animation """
                     if scale_for_animation and (frame_num == 1 or frame_num == 4): 
                          input_name = examples[0]['input']
                          filename = input_name.split('/')[-1]
                          filename = filename.split('.')[0:-1]
                          filename = '.'.join(filename)
                          low_crop = 0.3; high_crop = 0.7; 
                          z_crop_h = 0.6
                          
                          
                          """ Skip if not within middle crop"""
                          #if np.min(cell.coords[:, 0]) < track_cur_seg.shape[0] * low_crop or np.max(cell.coords[:, 0]) > track_cur_seg.shape[0] * high_crop or  np.min(cell.coords[:, 1]) < track_cur_seg.shape[1] * low_crop or np.max(cell.coords[:, 1]) > track_cur_seg.shape[1] * high_crop or np.max(cell.coords[:, 2]) > track_cur_seg.shape[2] * z_crop_h:
                              
                          if cell.X > track_cur_seg.shape[0] * low_crop and cell.X < track_cur_seg.shape[0] * high_crop and cell.Y > track_cur_seg.shape[1] * low_crop and cell.Y < track_cur_seg.shape[1] * high_crop and cell.Z < track_cur_seg.shape[2] * z_crop_h:
                              ### PLOT current and next frame
                              #plot_next = np.copy(next_seg)
                              #plot_next[plot_next != 250] = 0     ### delete everything that hasn't been visited
                              #plot_next[plot_next > 0] = 150      ### MAKE THE BACKGROUND PAST TRACES DIMMER
                              from random import randint
                              rand = randint(1, 6)
                              if new:  ### if a cell was tracked on second frame, use those coords to highlight it brighter than the rest
                                   track_new_seg = np.zeros(np.shape(track_cur_seg))
                                   track_new_seg[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = 255
                                  
                                   track_new_seg_next = np.zeros(np.shape(track_cur_seg))
                                   track_new_seg_next[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 255       
                                  
                                  
                                   # track_cur_seg[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = rand
                                   # plot_next[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = rand    
                                   copy_next_input[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 0
                                  
                              else:   ### otherwise, if NOT TRACKED, then change the color of the cell on the FIRST FRAME!!!
                                  
                                  ### still have to move the magenta dot!!!
                                  track_new_seg = np.zeros(np.shape(track_cur_seg))
                                  track_new_seg[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = 255
                                   
                                  ### and set the magenta dot on the next frame to be empty
                                  track_new_seg_next = np.zeros(np.shape(track_cur_seg))
    
                                  #track_cur_seg[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = 0
                                  
                        
                                  
                              """ Print out animation for 2nd frame """
                              #copy_next_input[plot_next > 0] = 0      ### set old cells to blank so color comes through better!
                              
                            
                              im = convert_matrix_to_multipage_tiff(copy_next_input)
                              im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                              im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation), order = 0)
                              imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num) +  '_cell_num_' + str(cell_idx) + '_next_input.tif',  np.asarray(im * 255, dtype=np.uint8))
                              
                              im = convert_matrix_to_multipage_tiff(plot_next)
                              im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                              im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation), order = 0)
                              imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num) +  '_cell_num_' + str(cell_idx) + '_next_seg.tif',  np.asarray(im * 255, dtype=np.uint8))
    
    
                              im = convert_matrix_to_multipage_tiff(track_new_seg_next)
                              im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                              im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation))
                              imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num) +  '_cell_num_' + str(cell_idx) + '_next_CUR_CHECK.tif',  np.asarray(im * 255, dtype=np.uint8))
                               
                              
    
                              """ Print out animation for 1st frame """
                              copy_input_im[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = 0   ### set old cells to blank so color comes through better!
                              im = convert_matrix_to_multipage_tiff(copy_input_im)
                              
                              ### or just crop it
                              im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                              im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation), order = 0)
                              imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num - 1) + '_cell_num_' + str(cell_idx) + '_cur_input.tif',  np.asarray(im * 255, dtype=np.uint8))
            
                              im = convert_matrix_to_multipage_tiff(track_cur_seg)
                              im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                              im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation), order = 0)
                              imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num - 1) +  '_cell_num_' + str(cell_idx) + '_cur_seg.tif',  np.asarray(im * 255, dtype=np.uint8))
        
                              im = convert_matrix_to_multipage_tiff(track_new_seg)
                              im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                              im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation))
                              imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num - 1) +  '_cell_num_' + str(cell_idx) + '_cur_CURRENT_seg.tif',  np.asarray(im * 255, dtype=np.uint8))
        
        
                              im = convert_matrix_to_multipage_tiff(track_term_seg)
                              im = im[0 : int(im.shape[0] * z_crop_h), int(im.shape[1] * low_crop) : int(im.shape[1] * high_crop),  int(im.shape[2] * low_crop) : int(im.shape[2] * high_crop)]
                              im = resize(im, (im.shape[0], im.shape[1] * scale_for_animation, im.shape[2]  * scale_for_animation), order = 0)
                              imsave(sav_dir + filename + '_ANIMATION_iterator_' + str(animator_iterator) + '_frame_num_' + str(frame_num - 1) +  '_cell_num_' + str(cell_idx) + '_cur_TERM.tif',  np.asarray(im * 255, dtype=np.uint8))
        
        
                              animator_iterator += 1
                              
                              
                              ### add rainbow color index AFTER iterating through current index
                              if new:
                                   track_cur_seg[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = rand
                                   plot_next[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = rand
                                   
                              else:
                                  track_cur_seg[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = rand
                                  track_term_seg[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = 255
                                   
                              # if animator_iterator == 5:
                              #     zzz
                                 
                              print('YOOOOOOOOOOO')




            """ POST-PROCESSING on per-frame basis """   
            all_size_pairs.append(size_pairs)


            if len(tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]].X) > 0:
                              print('pause')        
                              
            
            """ #1 == check all cells with tracked predictions """
            tmp = tracked_cells_df.copy()
            tmp_next_seg = np.copy(next_seg)
            
 
            ### DEBUG:
            tracked_cells_df = tmp.copy()
            next_seg = np.copy(tmp_next_seg)
            
            """  Identify all potential errors of tracking (over the predicted distance error threshold) """
            tracked_cells_df, all_dist, dist_check, check_series = check_predicted_distances(tracked_cells_df, frame_num, crop_size, z_size, dist_error_thresh=10)
            if truth:
                plt.figure(); plt.hist(all_dist)
            
            
            """ Keep looping above and start searnchig from LARGEST distance cells to correct """
            concat = np.transpose(np.asarray([check_series, dist_check]))
            sortedArr = concat[concat[:,1].argsort()[::-1]]
            check_sorted_series = sortedArr[:, 0]
            
            ### ^^^this order is NOT being kept right now!!!
             

            """ Check all cells with distance that is going against prediction """
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, check_sorted_series,
                                                                                next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                width_tmp, depth_tmp, input_im=input_im, 
                                                                                next_input=next_input, cur_seg=cur_seg,
                                                                                min_dist=10)
            
            if len(tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]].X) > 0:
                              print('pause')    
                              
                              
            new_candidates = recheck_series
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, new_candidates,   
                                                                                next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                width_tmp, depth_tmp, input_im=input_im,
                                                                                next_input=next_input, cur_seg=cur_seg,
                                                                                min_dist=10)            
            if len(tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]].X) > 0:
                              print('pause')    
                              

            """ #2 == Check duplicates """
            all_dup_indices = np.unique(all_dup_indices)
            
            
            # dup_overall = tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]]
            ### only keep what's hasn't been deleted above
            
            #l3 = [x for x in l1 if x not in l2]
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, all_dup_indices,   
                                                                                next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                width_tmp, depth_tmp, input_im=input_im, 
                                                                                next_input=next_input, cur_seg=cur_seg,
                                                                                min_dist=10)


            if len(tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]].X) > 0:
                              print('pause')    

            ### check on recheck_series
            recheck_series = np.unique(recheck_series)
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, recheck_series,   
                                                                                next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                width_tmp, depth_tmp, input_im=input_im, 
                                                                                next_input=next_input, cur_seg=cur_seg,
                                                                                min_dist=10)
            """ #3 == check all terminated cells """
            all_cur_frame = np.where(tracked_cells_df["FRAME"] == frame_num - 1)[0]
            cur_series = tracked_cells_df.iloc[all_cur_frame].SERIES
            term_series = []
            for cur in cur_series:
                index_cur = np.where((tracked_cells_df["SERIES"] == cur) & (tracked_cells_df["FRAME"] == frame_num - 1))[0]
                index_next = np.where((tracked_cells_df["SERIES"] == cur) & (tracked_cells_df["FRAME"] == frame_num))[0]
                 
                """ if next frame is empty, then terminated """
                if len(index_next) == 0:
                    term_series.append(cur)




            if len(tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]].X) > 0:
                              print('pause')      


            term_series = np.concatenate((term_series, recheck_series))
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, term_series,   
                                                                                next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                width_tmp, depth_tmp, input_im=input_im, 
                                                                                next_input=next_input, cur_seg=cur_seg,


                                                                                min_dist=10)

            ### then also checkup on all the following cells that were deleted and need to be rechecked
            tracked_cells_df, all_dist, dist_check, check_series = check_predicted_distances(tracked_cells_df, frame_num, crop_size, z_size, dist_error_thresh=10)
            if truth:
                plt.figure(); plt.hist(all_dist)          

            ### check on recheck_series
            recheck_series = np.concatenate((check_series, np.unique(recheck_series)))
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, recheck_series,   
                                                                                next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                width_tmp, depth_tmp, input_im=input_im, 
                                                                                next_input=next_input, cur_seg=cur_seg,
                                                                                min_dist=10)

            ### then also checkup on all the following cells that were deleted and need to be rechecked
            tracked_cells_df, all_dist, dist_check, check_series = check_predicted_distances(tracked_cells_df, frame_num, crop_size, z_size, dist_error_thresh=10)
            if truth:
                plt.figure(); plt.hist(all_dist)  
  


            """ Reverse and check each "NEW" cell to ensure it's actually new """
            size_ex = 100
            size_upper = 1000   ### above this is pretty certain will be a large cell so no need to test
            bw_next_seg = np.copy(next_seg)
            bw_next_seg[bw_next_seg > 0] = 1
            
            labelled = measure.label(bw_next_seg)
            next_cc = measure.regionprops(labelled)
            
            ### add the cells from the first frame into "tracked_cells" matrix
            num_new = 0; num_new_truth = 0

            """ Get true cur_seg, of what has been LINKED already """
            cur_seg_LINKED = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num - 1, input_im=input_im)


            for idx, cell in enumerate(next_cc):
               coords = cell['coords']
               
               if not np.any(next_seg[coords[:, 0], coords[:, 1], coords[:, 2]] == 250) and (len(coords) > size_ex and len(coords) < size_upper):   ### 250 means already has been visited    
                    centroid = cell['centroid']
                    #print(len(coords))
                    
                    ### go to unvisited cells
                    x = int(centroid[0]); y = int(centroid[1]); z = int(centroid[2]);
                         
                    """ Crop and prep data for CNN, ONLY for the first 2 frames so far"""
                    batch_x, crop_im, crop_cur_seg, crop_seed, crop_next_input, crop_next_seg, crop_next_seg_non_bin, box_xyz, box_over = prep_input_for_CNN(cell, next_input, input_im, next_seg,
                                                                                                             cur_seg_LINKED, mean_arr, std_arr, x, y, z, crop_size, z_size,
                                                                                                             height_tmp, width_tmp, depth_tmp, next_bool=next_bool)
    
                    ### Convert to Tensor
                    inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)
    
                    # forward pass to check validation
                    output_val = unet(inputs_val)
    
                    """ Convert back to cpu """                                      
                    output_val = np.moveaxis(output_val.cpu().data.numpy(), 1, -1)      
                    seg_train = np.moveaxis(np.argmax(output_val[0], axis=-1), 0, -1)
    
                    ### plot
                    debug = 0
                    if debug:
                        plot_max(crop_im, ax=-1)
                        plot_max(crop_cur_seg, ax=-1)
                        plot_max(crop_next_input, ax=-1)
                
                        crop_next_copy = np.copy(crop_next_seg_non_bin)
                
                        crop_next_copy[crop_next_copy == 250] = 1
                        crop_next_copy[crop_next_copy == 255] = 2
                     
                        plot_max(crop_next_copy, ax=-1)
                        plot_max(seg_train, ax=-1)
                        print('debug')
                        
                  
                    """ (1) if identified something in seg_train, then this is NOT a new cell!!!
                                - then need to check with cur_seg to see if it exists
                                (a) if it has NOT been identified before, then set it as such ==> add entirely new series
                                (b) if it HAS been identified before, then this is still likely new cell
                    
                    """
                    label = measure.label(seg_train)
                    cc_seg_train = measure.regionprops(label)
                    
                       
                    for cc in cc_seg_train:
                      cur_coords = cc.coords
                      
                      ### check if matched or not???
                      if not np.any(crop_next_seg_non_bin[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] > 0) and len(cur_coords) > min_size:  ### don't want single pixel stuff either
                          cur_coords = scale_coords_of_crop_to_full(cur_coords, box_xyz, box_over)
                          
                          cur_centroid = np.asarray(cc_seg_train[0].centroid)
                          cur_centroid = scale_single_coord_to_full(cur_centroid, box_xyz, box_over)
                          
                          
                          """ Need to check to ensure the coords do not go over the image size limit because the CNN output is NOT subjected to the limits!!! """
                          cur_coords = check_limits([cur_coords], width_tmp, height_tmp, depth_tmp)[0]
                          cur_centroid = check_limits_single([cur_centroid], width_tmp, height_tmp, depth_tmp)[0]
    
                          """ Create new entry for cell on CURRENT FRAME and put into series """
                          ### add to matrix
                          series = np.max(tracked_cells_df.SERIES) + 1
                          
                          row = {'SERIES': series, 'COLOR': 'GREEN', 'FRAME': frame_num - 1, 'X': int(cur_centroid[0]), 'Y':int(cur_centroid[1]), 'Z': int(cur_centroid[2]), 'coords':cur_coords, 'visited': 1}
                          tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)     



                          """ Then create next entry for cell in NEXT FRAME """
                          row = {'SERIES': series, 'COLOR': 'GREEN', 'FRAME': frame_num, 'X': int(x), 'Y':int(y), 'Z': int(z), 'coords':coords, 'visited': 0}
                          tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)     



                          cur_seg_LINKED[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = 1

                          if len(tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]].X) > 0:
                              print('pause')                          
                          
                          """ set current one to be value 2 so in future will know has already been identified """
                          next_seg[coords[:, 0], coords[:, 1], coords[:, 2]] = 250;  
                          
                          
                          
                          break;   ### HACK: currently just goes with the first object that is NOT associated with something in a previously linked cell   
                          
                          
                      
                    """ (2) if NOTHING identified in seg_train, then possible candidate for being a NEW CELL!!! go down below for more size exclusion  
                    
                    """                    
                    
                    """ Fixes:
                            - too small
                            - doubles
                            - check if ACTUALLY has been there before, if not, still okay!!!
                        
                        """
                          ### update cur_seg_linked:
                                                
                              
            if len(tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]].X) > 0:
                to_recheck = tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]].index
                print(to_recheck)
                print('pause')
                to_recheck = np.asarray(to_recheck)
                tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, recheck_series,   
                                                                                    next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                    width_tmp, depth_tmp, input_im=input_im, 
                                                                                    next_input=next_input, cur_seg=cur_seg,
                                                                                    min_dist=10)                              
                          
                
                
            
            """ associate remaining cells that are "new" cells and add them to list to check as well as the TRUTH tracker """
            new_min_size = 250
            if not truth:
                truth_next_im = 0
            tracked_cells_df, truth_output_df, truth_next_im, truth_array = associate_remainder_as_new(tracked_cells_df, next_seg, frame_num, lowest_z_depth, z_size, min_size=new_min_size,
                                                                           truth=truth, truth_output_df=truth_output_df, truth_next_im=truth_next_im, truth_array=truth_array)


            if len(tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]].X) > 0:
                to_recheck = tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]].index
                print(to_recheck)
                print('pause')
                
                ### HACK:
                """ HACK: somewhere next_seg is not being set properly, so some cells are being marked as NEW even though they've already been associated before """
                for redo in to_recheck:
                    cell = tracked_cells_df.loc[redo]
                    series_redo = cell.SERIES
         
                    num_frames = len(tracked_cells_df.iloc[np.where((tracked_cells_df.SERIES == series_redo))[0]])
                    
                    print(num_frames)
                    if num_frames == 1:  ### DROP
                         tracked_cells_df = tracked_cells_df.drop(redo)   
                
                to_recheck = tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]].index
                if len(to_recheck) > 0:
                    print(to_recheck)
                    print('pause')                
                
                
                

                             
            debug_seg = np.copy(next_seg)
            debug_seg[debug_seg == 255] = 1 
            debug_seg[debug_seg == 250] = 2 
            
            
            
            
                    
            """ delete any 100% duplicated rows
                    ***figure out WHY they are occuring??? probably from re-checking???
                    
                    
                    DOUBLE CHECK THIS HACK???
            """
            tmp_drop = tracked_cells_df.copy()
            tmp_drop = tmp_drop.drop(columns='coords')
            dup_idx = np.where(tmp_drop.duplicated(keep="first")) ### keep == True ==> means ONLY tells you subseqent duplicates!!!
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[dup_idx])
            
            
            
            """ Set next frame to be current frame """
            ### for debug
            tmp_cur = np.copy(cur_seg)
            tmp_input = np.copy(input_im)
            #plot_max(next_seg, ax=-1)
            
            input_im = next_input
            
            if scale_for_animation:
                copy_input_im = np.copy(next_input)
                
            
            cur_seg = next_seg
            cur_seg[cur_seg > 0] = 255    ### WAS WRONGLY SET TO 0 BEFORE!!!
            truth_cur_im = truth_next_im
            
            
            
    """ DONT DO SIZE ELIM WITHIN LOOP, DO AS POST-PROCESSING INSTEAD???
    
            - this way removes entire track if there's even one part that's too small
    """
            
    

    """
        Things to do:
            (1) find out which cells being eliminated and why
            (2) fix few duplicates left
            (3) fix XY coordinates
            (4) fix large blobs being segmented together
    
            (5) mark "new" cells to ensure they are "new"
            
            
            (6) looking at density ==> layer 2 maybe partition into different layers?
                - so that layer 2 isn't taking into account cells from lower/upper layers?
    
            (7) ***plot to see how well you can predict next cell location!!! based on surroundings!!!!
                            *** use truth_array???
    
    """
    
            
          
            
    """ Plot size pairs from inference """
    # for pair_idx, frame_pairs in enumerate(all_size_pairs):
    #     plt.figure(); plt.title(str(pair_idx))
    #     print(len(frame_pairs))
    #     for pair in frame_pairs:
            
    #         if pair[1] > 1000 and pair[0] < 500:
    #             plt.plot(pair); 
    #     plt.ylim([0, 3000]); 
            
            
            
            
    """ POST-PROCESSING """
            
    """ Parse the old array and SAVE IT: """
    print('duplicates: ' + str(np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME']))))    ### *** REAL DUPLICATES
    tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]]
    
    print('double_linked throughout analysis: ' + str(double_linked))
    #print('num_blobs: ' + str(num_blobs))
    #print('duplicates: ' + str(np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'SERIES']))))   ### cell in same location across frames
    #tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'SERIES'], keep=False))[0]]

    
    ### (1) unsure that all of 'RED' or 'YELLOW' are indicated as such
    ### ***should be fine, just turn all "BLANK" into "GREEN"  
    tracked_cells_df.COLOR[tracked_cells_df['COLOR'] == 'BLANK'] = 'Green'
    tracked_cells_df.COLOR[tracked_cells_df['COLOR'] == 'GREEN'] = 'Green'
    
    num_YELLOW = 0; num_RED = 0; num_new_color = 0;
    for cell_num in np.unique(tracked_cells_df.SERIES):
       
        color_arr = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].COLOR)
        
        cells = tracked_cells_df.loc[tracked_cells_df["SERIES"].isin([cell_num])]
        index = cells.index
        
        if np.any(color_arr == 'RED'):
            tracked_cells_df["COLOR"][index] = 'RED'
            num_RED += 1
            
        elif np.any(color_arr == 'RED'):  # originally YELLOW
            tracked_cells_df["COLOR"][index]= 'RED'
            num_YELLOW += 1
            
        ### Mark all new cells on SINGLE FRAME as 'YELLOW'
        if len(color_arr) == 1:
            tracked_cells_df["COLOR"][index] = 'BLUE'
            num_new_color += 1
            #print(color_arr)
            
            
    
    tracked_cells_df.COLOR[tracked_cells_df['COLOR'] == 'YELLOW'] = 'Yellow' 
    tracked_cells_df.COLOR[tracked_cells_df['COLOR'] == 'RED'] = 'Red' 


    
    """ Get new cells and terminated cells ONLY from a dataframe """
    """ Mark all new cells as BLUE """
    
    
    ### CANT DO THIS BECAUSE EVERY CELL MUST HAVE ONLY 1 color
    # for frame_num, im_dict in enumerate(examples):
    #     for idx_truth in np.where(tracked_cells_df.FRAME == frame_num)[0]:
            
    #         ### get only NEWLY FORMED cells, excluding if on 1st frame
    #         if frame_num > 0:
    #             cell = tracked_cells_df.iloc[idx_truth]
    #             series = cell.SERIES
                
    #             prev_cell = np.where((tracked_cells_df.FRAME == frame_num - 1) & (tracked_cells_df.SERIES == series))[0]

                
    #             if len(prev_cell) == 0:   ### MEANS NEWLY FORMED
    #                 #print(tracked_cells_df.iloc[idx_truth].SERIES)
    #                 print(frame_num)
    #                 zzz
                    

    
    
        

    ### (4) re-name X and Y columns
    tracked_cells_df = tracked_cells_df.rename(columns={'X': 'Y', 'Y': 'X'})


    
    """ Pre-save everything """
    tracked_cells_df = tracked_cells_df.sort_values(by=['SERIES', 'FRAME'])
    tracked_cells_df.to_csv(sav_dir + 'tracked_cells_df_RAW.csv', index=False)
    
    tracked_cells_df.to_pickle(sav_dir + 'tracked_cells_df_RAW_pickle.pkl')
    
    #tracked_cells_df = pd.read_pickle(sav_dir + 'tracked_cells_df_RAW_pickle.pkl')
    
    ### (2) remove everything only on a single frame, except for very first frame
    singles = []
    deleted = 0
    for cell_num in np.unique(tracked_cells_df.SERIES):
          
               track_length_SEG = len(np.unique(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME))         
               """ remove anything that's only tracked for length of 1 timeframe """
               """ excluding if that timeframe is the very first one OR the very last one"""
        
               if len(np.where(np.asarray(track_length_SEG) == 1)[0]) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == 0) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == np.max(tracked_cells_df.FRAME)):
                           singles.append(cell_num)
                           tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[np.where(tracked_cells_df.SERIES == cell_num)])
                           #print(cell_num)
                           
                           deleted += 1
                           continue;
                           
                           
                           
    """ (3) ALSO clean up bottom of image so that no new cell can appear in the last 20 stacks
                also maybe remove cells on edges as well???
    """
    num_edges = 0;
    num_bottom = 0;
    
    for cell_num in np.unique(tracked_cells_df.SERIES):
        idx = np.where(tracked_cells_df.SERIES == cell_num)[0]
        Z_cur_cell = tracked_cells_df.iloc[idx].Z
        X_cur_cell = tracked_cells_df.iloc[idx].X
        Y_cur_cell = tracked_cells_df.iloc[idx].Y

        if np.any(Z_cur_cell > lowest_z_depth - 20):
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])
            
            num_bottom += 1
            
        elif np.any(X_cur_cell > width_tmp - exclude_side_px) or np.any(X_cur_cell < exclude_side_px):
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])
            num_edges += 1
            
        
        elif np.any(Y_cur_cell > height_tmp - exclude_side_px) or np.any(Y_cur_cell < exclude_side_px):
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])
            
            num_edges += 1
                

    """ Also remove by min_size 
    
    
                ***ONLY IF SMALL WITHIN FIRST FEW FRAMES???
    
    """
    num_small = 0; real_saved = 0; upper_thresh = 500;  lower_thresh = 400; small_start = 0;
    for cell_num in np.unique(tracked_cells_df.SERIES):
                
        idx = np.where(tracked_cells_df.SERIES == cell_num)
        all_lengths = []
        small_bool = 0;
        for iter_idx, cell_obj in enumerate(tracked_cells_df.iloc[idx].coords):
            
            
            start_frame = np.asarray(tracked_cells_df.iloc[idx].FRAME)[0]
            
            if len(cell_obj) < min_size:  
                small_bool = 1


            ### if start is super small, then also delete
            if len(cell_obj) < lower_thresh and iter_idx == 0 and start_frame != 0:  
                small_bool = 1
                small_start += 1
            
            ### exception, spare if large cell within first frame
            if len(cell_obj) > upper_thresh and iter_idx < 1:  
                small_bool = 0
                real_saved += 1
                break;
        
        if small_bool:
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])   ### DROPS ENTIRE CELL SERIES
            num_small += 1

                
    """  Save images in output """
    input_name = examples[0]['input']
    filename = input_name.split('/')[-1]
    filename = filename.split('.')[0:-1]
    filename = '.'.join(filename)
    
    for frame_num, im_dict in enumerate(examples):
         
            output_frame = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im)
            im = convert_matrix_to_multipage_tiff(output_frame)
            imsave(sav_dir + filename + '_' + str(frame_num) + '_output_CLEANED.tif', im)
         
         
              # output_frame = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, color=1)
              # im = convert_matrix_to_multipage_tiff(output_frame)
              # imsave(sav_dir + filename + '_' + str(frame_num) + '_output_COLOR.tif', im)
    

            # output_frame = gen_im_new_term_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, new=0)
            # im = convert_matrix_to_multipage_tiff(output_frame)
            # imsave(sav_dir + filename + '_' + str(frame_num) + '_output_TERMINATED.tif', im)


            output_frame = gen_im_new_term_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, new=1)
            im = convert_matrix_to_multipage_tiff(output_frame)
            imsave(sav_dir + filename + '_' + str(frame_num) + '_output_NEW.tif', im)
            
            
            
            
    """ Drop unsaveable stuff and re-order the axis"""
    tracked_cells_df = tracked_cells_df.drop(['coords', 'visited'], axis=1)
   
    cols = ['SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z']
           
    tracked_cells_df = tracked_cells_df[cols]
            
    tracked_cells_df.to_csv(sav_dir + 'tracked_cells_df_clean.csv', index=False)            
            
            
            
            
            

    """ Set globally """
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    plt.rcParams['figure.dpi'] = 300
    ax_title_size = 18
    leg_size = 14

    """ plot timeframes """
    norm_tots_ALL, norm_new_ALL = plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_', depth_lim_lower=0, depth_lim_upper=120, ax_title_size=ax_title_size, leg_size=leg_size)
    
    """ 
        Also split by depths
    """
    norm_tots_32, norm_new_32 = plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_0-32', depth_lim_lower=0, depth_lim_upper=32, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
    norm_tots_65, norm_new_65 = plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_33-65', depth_lim_lower=33, depth_lim_upper=65, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
    norm_tots_99, norm_new_99 = plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_66-99', depth_lim_lower=66, depth_lim_upper=99, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
    norm_tots_132, norm_new_132 = plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_100-132', depth_lim_lower=100, depth_lim_upper=132, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
    norm_tots_165, norm_new_165 = plot_timeframes(tracked_cells_df, sav_dir, add_name='OUTPUT_133-165', depth_lim_lower=133, depth_lim_upper=165, only_one_plot=1, ax_title_size=ax_title_size, leg_size=leg_size)
                
            
         