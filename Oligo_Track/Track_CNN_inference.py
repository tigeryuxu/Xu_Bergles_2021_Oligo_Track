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
    
    
    
    ***NOTES FOR USER:
        - cells on border are dropped
        - cells at bottom are also dropped
            ***include in GUI
    
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

from skimage import measure
import pandas as pd

import tifffile as tiff
import tkinter
from tkinter import filedialog

import torch
# from torch import nn
# import torch.nn.functional as F
# import torch.optim as optim

from UNet_pytorch_online import *
# from PYTORCH_dataloader import *
from functional.UNet_functions_PYTORCH import *

from functional.matlab_crop_function import *
from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.functions_cell_track_auto import *
from skimage.transform import rescale, resize, downscale_local_mean


from plot_functions.PLOT_FIGURES_functions import *


# import pandas as pd
# import scipy.stats as sp
# import seaborn as sns


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
s_path = './(10) Checkpoints_full_auto_no_spatialW_large_TRACKER_CROP_PADS_NO_NEXT_only_check/'; next_bool = 0;


lowest_z_depth = 140;

crop_size = 160; z_size = 32; num_truth_class = 2
min_size = 10
elim_size = 100; exclude_side_px = 40

min_size = 100;  upper_thresh = 800;

scale_xy = 0.83; scale_z = 3

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
mean_arr = tracker.mean_arr; std_arr = tracker.std_arr

unet = check['model_type']; unet.load_state_dict(check['model_state_dict'])
unet.to(device); unet.eval()


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
    sav_dir = input_path + '/' + foldername + '_output_FULL_AUTO_no_next_10_125762_TEST_8_INTENSITY'

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
    
    """ Load input data and segmentations for Seg-CNN"""   
    input_name = examples[0]['input']
    input_im = tiff.imread(input_name);  depth_total, empty, empty = input_im.shape
    input_im = np.moveaxis(input_im, 0, -1); width_tmp, height_tmp, depth_tmp = input_im.shape
    
    ### get Seg-CNN outputs
    seg_name = examples[0]['seg']  
    cur_seg = tiff.imread(seg_name)
    cur_seg[lowest_z_depth:-1, ...] = 0   ### CLEAR EVERY CELL BELOW THIS DEPTH
    cur_seg = np.moveaxis(cur_seg, 0, -1)
    
    
    """  add the cells from the first frame into "tracked_cells" matrix
    """
    cur_seg[cur_seg > 0] = 1
    labelled = measure.label(cur_seg)
    cur_cc = measure.regionprops(labelled, intensity_image=input_im)
    tracked_cells_df = pd.DataFrame(columns = {'SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z', 'coords', 'visited', 'mean_intensity'})     
    
    for cell in cur_cc:
         if not np.isnan(np.max(tracked_cells_df.SERIES)):
              series = np.max(tracked_cells_df.SERIES) + 1
         else:
                   series = 1
         centroid = cell['centroid']
         coords = cell['coords']
         intensity = cell['mean_intensity']
         
         """ DONT TEST IF TOO SMALL """
         if len(coords) < min_size:
              continue;         
         row = {'SERIES': series, 'COLOR': 'BLANK', 'FRAME': 0, 'X': int(centroid[0]), 'Y':int(centroid[1]), 'Z': int(centroid[2]), 
                'coords':coords, 'mean_intensity':intensity, 'visited': 0}
         tracked_cells_df = tracked_cells_df.append(row, ignore_index=True) 
    
    """ Start looping through segmented nuclei """
    animator_iterator = 0;
    all_size_pairs = []
    for frame_num in range(1, len(examples)):
         print('Starting inference on volume: ' + str(frame_num) + ' of total: ' + str(len(examples)))
         all_dup_indices = [];
        
         with torch.set_grad_enabled(False):  # saves GPU RAM            

            """ Gets next seg as well """
            input_name = examples[frame_num]['input']            
            next_input = tiff.imread(input_name);
            next_input = np.moveaxis(next_input, 0, -1)

            
            seg_name = examples[frame_num]['seg']
            next_seg = tiff.imread(seg_name);
            next_seg[lowest_z_depth:-1, ...] = 0   ### CLEAR EVERY CELL BELOW THIS DEPTH
            next_seg = np.moveaxis(next_seg, 0, -1)


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


                 """ ***IF MORE THAN ONE OBJECT IS IN FINAL SEGMENTATION, choose the best matched one!
                 """
                 cc_seg_train, seg_train = select_one_from_excess(seg_train, crop_next_seg, crop_next_input)
                 
                      
                 """ if the next segmentation is EMPTY OR if it's tiny, just skip it b/c might be error noise output """
                 if len(cc_seg_train) == 0 or len(cc_seg_train[0].coords) < 20:                      
                      debug = 0                      
                 else:
                                          
                     """ Check if current volume is much smaller than ENORMOUS next volume """
                     size_pairs.append([len(cell.coords), len(cc_seg_train[0].coords)])
                     #if len(cell.coords) < 500 and len(cc_seg_train[0].coords) > 1000:
                     if (len(cell.coords) > 750 or len(cc_seg_train[0].coords) > 1500) and len(cc_seg_train[0].coords) > len(cell.coords) * 2:
                            cc_seg_train = []
                            
       
                     """ Find coords of identified cell and scale back up, later find which ones in next_seg have NOT been already identified
                     """
                     new = 0
                     if len(cc_seg_train) > 0:
                          next_coords = cc_seg_train[0].coords  
                          intensity = cc_seg_train[0]['mean_intensity']
                          
        
                          next_coords = scale_coords_of_crop_to_full(next_coords, box_xyz, box_over)
                          
                          next_centroid = np.asarray(cc_seg_train[0].centroid)
                          next_centroid = scale_single_coord_to_full(next_centroid, box_xyz, box_over)
                          
                          
                          """ Need to check to ensure the coords do not go over the image size limit because the CNN output is NOT subjected to the limits!!! """
                          next_coords = check_limits([next_coords], width_tmp, height_tmp, depth_tmp)[0]
                          next_centroid = check_limits_single([next_centroid], width_tmp, height_tmp, depth_tmp)[0]
    
    
                          """ FIND DOUBLES EARLY TO CORRECT AS YOU GO """
                          if np.any(next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] == 250): ### if this place has already been visited in the past
                               #print('double_linked'); 
                               #double_linked += 1                    
                               
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
                              row = {'SERIES': cell.SERIES, 'COLOR': 'GREEN', 'FRAME': frame_num, 'X': int(next_centroid[0]), 'Y':int(next_centroid[1]), 'Z': int(next_centroid[2]),
                                     'coords':next_coords, 'mean_intensity':intensity, 'visited': 0}
                              tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)     

                              """ set current one to be value 2 so in future will know has already been identified """
                              next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 250;  
                              new = 1;
                          
                     else:                           
                          ####   DEBUG if not matched
                          len(np.where(crop_seed)[0])         
                          
                          
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

            ### check on recheck_series
            recheck_series = np.concatenate((check_series, np.unique(recheck_series)))
            tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, recheck_series,   
                                                                                next_seg, crop_size, z_size, frame_num, height_tmp, 
                                                                                width_tmp, depth_tmp, input_im=input_im, 
                                                                                next_input=next_input, cur_seg=cur_seg,
                                                                                min_dist=10)

            ### then also checkup on all the following cells that were deleted and need to be rechecked
            tracked_cells_df, all_dist, dist_check, check_series = check_predicted_distances(tracked_cells_df, frame_num, crop_size, z_size, dist_error_thresh=10)



            """ Reverse and check each "NEW" cell to ensure it's actually new """
            size_ex = 100
            size_upper = 1000   ### above this is pretty certain will be a large cell so no need to test
            bw_next_seg = np.copy(next_seg)
            bw_next_seg[bw_next_seg > 0] = 1
            
            labelled = measure.label(bw_next_seg)
            next_cc = measure.regionprops(labelled, intensity_image=input_im)
            
            ### add the cells from the first frame into "tracked_cells" matrix
            num_new = 0; num_new_truth = 0

            """ Get true cur_seg, of what has been LINKED already """
            cur_seg_LINKED = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num - 1, input_im=input_im)


            for idx, cell in enumerate(next_cc):
               coords = cell['coords']
               
               if not np.any(next_seg[coords[:, 0], coords[:, 1], coords[:, 2]] == 250) and (len(coords) > size_ex and len(coords) < size_upper):   ### 250 means already has been visited    
                    centroid = cell['centroid']
                    intensity = cell['mean_intensity']
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
                           
                  
                    """ (1) if identified something in seg_train, then this is NOT a new cell!!!
                                - then need to check with cur_seg to see if it exists
                                (a) if it has NOT been identified before, then set it as such ==> add entirely new series
                                (b) if it HAS been identified before, then this is still likely new cell
                    
                    """
                    label = measure.label(seg_train)
                    cc_seg_train = measure.regionprops(label, intensity_image=crop_next_input)
                    
                       
                    for cc in cc_seg_train:
                      cur_coords = cc.coords
                      
                      ### check if matched or not???
                      if not np.any(crop_next_seg_non_bin[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] > 0) and len(cur_coords) > min_size:  ### don't want single pixel stuff either
                          cur_coords = scale_coords_of_crop_to_full(cur_coords, box_xyz, box_over)
                          
                          cur_centroid = np.asarray(cc_seg_train[0].centroid)
                          cur_centroid = scale_single_coord_to_full(cur_centroid, box_xyz, box_over)
                          
                          cur_intensity = np.asarray(cc_seg_train[0].mean_intensity)
                          
                          
                          """ Need to check to ensure the coords do not go over the image size limit because the CNN output is NOT subjected to the limits!!! """
                          cur_coords = check_limits([cur_coords], width_tmp, height_tmp, depth_tmp)[0]
                          cur_centroid = check_limits_single([cur_centroid], width_tmp, height_tmp, depth_tmp)[0]
    
                          """ Create new entry for cell on CURRENT FRAME and put into series """
                          ### add to matrix
                          series = np.max(tracked_cells_df.SERIES) + 1
                          
                          row = {'SERIES': series, 'COLOR': 'GREEN', 'FRAME': frame_num - 1, 'X': int(cur_centroid[0]), 'Y':int(cur_centroid[1]), 'Z': int(cur_centroid[2]), 
                                 'coords':cur_coords, 'mean_intensity':cur_intensity, 'visited': 1}
                          tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)     



                          """ Then create next entry for cell in NEXT FRAME """
                          row = {'SERIES': series, 'COLOR': 'GREEN', 'FRAME': frame_num, 'X': int(x), 'Y':int(y), 'Z': int(z), 
                                 'coords':coords, 'mean_intensity':intensity, 'visited': 0}
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
            truth = 0
            if not truth:
                truth_next_im = 0; truth_output_df = []; truth_array = [];
            tracked_cells_df, truth_output_df, truth_next_im, truth_array = associate_remainder_as_new(tracked_cells_df, next_seg, frame_num, lowest_z_depth, z_size, next_input, min_size=new_min_size,
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
            tmp_drop = tmp_drop.drop(columns='mean_intensity')
            dup_idx = np.where(tmp_drop.duplicated(keep="first")) ### keep == True ==> means ONLY tells you subseqent duplicates!!!
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[dup_idx])
            
            
            
            """ Set next frame to be current frame """
            ### for debug
            tmp_cur = np.copy(cur_seg)
            tmp_input = np.copy(input_im)
            input_im = next_input
            
            cur_seg = next_seg
            cur_seg[cur_seg > 0] = 255    ### WAS WRONGLY SET TO 0 BEFORE!!!
            truth_cur_im = truth_next_im
            
            
    
            

    """ POST-PROCESSING """
            
    """ Parse the old array and SAVE IT: """
    print('duplicates: ' + str(np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME']))))    ### *** REAL DUPLICATES
    tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]]
    
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
            tiff.imsave(sav_dir + filename + '_' + str(frame_num) + '_output_CLEANED.tif', im)
        

            output_frame = gen_im_new_term_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, new=1)
            im = convert_matrix_to_multipage_tiff(output_frame)
            tiff.imsave(sav_dir + filename + '_' + str(frame_num) + '_output_NEW.tif', im)
            

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
                
            
         