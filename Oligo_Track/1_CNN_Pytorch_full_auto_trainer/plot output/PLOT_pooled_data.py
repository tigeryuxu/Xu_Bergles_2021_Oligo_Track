#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:26:05 2020

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 00:47:35 2020

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

from matlab_crop_function import *
from plot_functions_CLEANED import *
from data_functions_CLEANED import *
from data_functions_3D import *
from functions_cell_track_auto import *
from skimage.transform import rescale, resize, downscale_local_mean


from PLOT_FIGURES_functions import *


import pandas as pd

import seaborn as sns



control = 0;

if control:
    lowest_z_depth = 120;
else:
    lowest_z_depth = 160;
    
crop_size = 160
z_size = 32
num_truth_class = 2
min_size = 10
both = 0


MATLAB = 0

scale_xy = 0.83
scale_z = 3
truth = 1


exclude_side_px = 40

min_size = 100;
upper_thresh = 800;




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

### TRACK WHICH FOLDERS TO POOL
folder_pools = np.zeros(len(list_folder))
all_norm_tots = []; all_norm_new = [];
all_norm_t32 = []; all_norm_n32 = [];
all_norm_t65 = []; all_norm_n65 = [];
all_norm_t99 = []; all_norm_n99 = [];
all_norm_t132 = []; all_norm_n132 = [];
all_norm_t165 = []; all_norm_n165 = [];


all_tracked_cells_df = [];

for fold_idx, input_path in enumerate(list_folder):
    foldername = input_path.split('/')[-2]
    #sav_dir = input_path + '/' + foldername + '_output_FULL_AUTO_no_next_10_125762_TEST_6'

    sav_dir = input_path + '/' + foldername + '_output_FULL_AUTO_no_next_10_125762_TEST_8_INTENSITY'



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
    
    
    
    """Load in uncleaned array  """
    tracked_cells_df = pd.read_pickle(sav_dir + 'tracked_cells_df_RAW_pickle.pkl')
    
    ### (2) remove everything only on a single frame, except for very first frame
    singles = []
    for cell_num in np.unique(tracked_cells_df.SERIES):
          
               track_length_SEG = len(np.unique(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME))         
               """ remove anything that's only tracked for length of 1 timeframe """
               """ excluding if that timeframe is the very first one OR the very last one"""
        
               if len(np.where(np.asarray(track_length_SEG) == 1)[0]) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == 0) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == np.max(tracked_cells_df.FRAME)):
               
               #if len(np.where(np.asarray(track_length_SEG) == 1)[0]) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == np.max(tracked_cells_df.FRAME)):
                           singles.append(cell_num)
                           tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[np.where(tracked_cells_df.SERIES == cell_num)])
                           #print(cell_num)
                           continue;
    """ (3) ALSO clean up bottom of image so that no new cell can appear in the last 20 stacks
                also maybe remove cells on edges as well???
    """
    num_edges = 0;
    
    for cell_num in np.unique(tracked_cells_df.SERIES):
        idx = np.where(tracked_cells_df.SERIES == cell_num)[0]
        Z_cur_cell = tracked_cells_df.iloc[idx].Z
        X_cur_cell = tracked_cells_df.iloc[idx].X
        Y_cur_cell = tracked_cells_df.iloc[idx].Y

        if np.any(Z_cur_cell > lowest_z_depth - 20):
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])
        elif np.any(X_cur_cell > width_tmp - exclude_side_px) or np.any(X_cur_cell < exclude_side_px):
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])
            num_edges += 1
            
        
        elif np.any(Y_cur_cell > height_tmp - exclude_side_px) or np.any(Y_cur_cell < exclude_side_px):
            tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[idx])
            
            num_edges += 1
                

    """ Also remove by min_size 
    
    
                ***ONLY IF SMALL WITHIN FIRST FEW FRAMES???
    
    """
    num_small = 0; real_saved = 0; upper_thresh = 500;  lower_thresh = 0; small_start = 0;
    for cell_num in np.unique(tracked_cells_df.SERIES):
                
        idx = np.where(tracked_cells_df.SERIES == cell_num)
        all_lengths = []
        small_bool = 0;
        for iter_idx, cell_obj in enumerate(tracked_cells_df.iloc[idx].coords):
            
            
            start_frame = np.asarray(tracked_cells_df.iloc[idx].FRAME)[0]
            
            if len(cell_obj) < min_size:  
                small_bool = 1


            ### if start is super small, then also delete, EXCLUDING START ON FIRST FRAME
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
         
            
         
            
    """ SCALE CELL COORDS to true volume 
    """
    tmp = np.zeros(np.shape(input_im))
    tracked_cells_df['vol_rescaled'] = np.nan
    print('scaling cell coords')
    for idx in range(len(tracked_cells_df)):
        
        cell = tracked_cells_df.iloc[idx]
        
        coords = cell.coords   
        tmp[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        
        crop, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(tmp, cell.X, cell.Y, cell.Z, 50/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
   
        crop_rescale = resize(crop, (crop.shape[0] * scale_xy, crop.shape[1] * scale_xy, crop.shape[2] * scale_z), order=0, anti_aliasing=True)
        
        label = measure.label(crop_rescale)       
        cc = measure.regionprops(label)
        new_coords = cc[0]['coords']
        tracked_cells_df.iloc[idx, tracked_cells_df.columns.get_loc('vol_rescaled')] = len(new_coords)
   
        tmp[tmp > 0] = 0  # reset
        
    """ Create copy """
    all_tracked_cells_df.append(tracked_cells_df)

            
                
    """ If filename contains 235 or 246, then must +1 to timeframes after baseline, because week 2 skipped """
    if '235' in foldername:
        
        for idx in range(len(tracked_cells_df)):
            
            cell = tracked_cells_df.iloc[idx]
            if cell.FRAME >= 1:
                new_val = cell.FRAME + 1
                cell.FRAME = new_val
                tracked_cells_df.iloc[idx] = cell
            
    if '264' in foldername:
        
        for idx in range(len(tracked_cells_df)):
            
            cell = tracked_cells_df.iloc[idx]
            if cell.FRAME >= 0:
                new_val = cell.FRAME + 3
                cell.FRAME = new_val
                tracked_cells_df.iloc[idx] = cell
            




    """ Normalize to 0 um using vertices """
    
    if '235' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        z_0 = 0; z_y = 27 - 2  ### upper right corner, y=0
        z_x = 0; z_xy = 31 - 2
        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)
        
        
        
        two_thirty_tracked = tracked_cells_df.copy()
        mesh_235 = np.copy(mesh)
        

    elif '037' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        z_0 = 18 - 0; z_y = 18 - 0   ### upper right corner, y=0
        z_x = 2 - 0; z_xy = 7 - 0
        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)    

    elif '030' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        
        z_0 = 16 + 2; z_y = 30 + 2   ### upper right corner, y=0
        z_x = 6 + 2; z_xy = 26 + 2

        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)  
        
        folder_pools[fold_idx] = 1;   ### TO POOL LATER

    elif '033' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        
        z_0 = 12 - 0; z_y = 21 - 0   ### upper right corner, y=0
        z_x = 5 - 0; z_xy = 16 - 0

        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)    

        folder_pools[fold_idx] = 1;   ### TO POOL LATER

    elif '097' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        
        z_0 = 10 + 1; z_y = 19 + 1   ### upper right corner, y=0
        z_x = 2 + 1; z_xy = 10 + 1

        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)   
        
        folder_pools[fold_idx] = 2;   ### TO POOL LATER

    elif '099' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        
        z_0 = 0 + 3; z_y = 15 + 3   ### upper right corner, y=0
        z_x = 0 + 3; z_xy = 15 + 3

        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)  
        
        folder_pools[fold_idx] = 2;   ### TO POOL LATER
        
        
        
        
        
        
    """ 
        Normalize for control experiments
    
    """
    if '056' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        z_0 = 10 + 3; z_y = 12 + 3  ### upper right corner, y=0
        z_x = 4 + 3; z_xy = 8 + 3
        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)
        
        control = 1;       
        

    elif '264' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        z_0 = 5 + 6; z_y = 48 + 6   ### upper right corner, y=0
        z_x = 0 + 6; z_xy = 32 + 6
        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)        
        
        control = 1;     
    
        ### pool 1
    elif '089' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        z_0 = 2 + 3; z_y = 2 + 3   ### upper right corner, y=0
        z_x = 2 + 3; z_xy = 2 + 3
        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)       
        
        folder_pools[fold_idx] = 1
        
        control = 1;     
        
    elif '115' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        z_0 = 11; z_y = 7   ### upper right corner, y=0
        z_x = 2; z_xy = 6
        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)       
        
        control = 1;     
        
        folder_pools[fold_idx] = 1
        
    elif '186' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        z_0 = 5 + 3; z_y = 5 + 3  ### upper right corner, y=0
        z_x = 5 + 3; z_xy = 5 + 3
        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)    
        
        control = 1;     
        folder_pools[fold_idx] = 1
        
        
        ### pool 2
    elif '385' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        z_0 = 4; z_y = 11   ### upper right corner, y=0
        z_x = 4; z_xy = 11
        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)    
        
        control = 1;     
        folder_pools[fold_idx] = 0
        
    elif '420' in foldername:
        x_0 = 0; y_lim = height_tmp
        y_0 = 0; x_lim = width_tmp
        z_0 = 4 + 2; z_y = 6 + 2   ### upper right corner, y=0
        z_x = 4 + 2; z_xy = 6 + 2
        mesh = points_to_mesh(x_0, x_lim, y_0, y_lim, z_0, z_y, z_x, z_xy)    
        
        control = 1;     
        folder_pools[fold_idx] = 2
        
        
        
        
    """ Scale Z to mesh """
    #m = np.max(input_im, axis=-1)
    print('scaling cell Z to mesh')
    for idx in range(len(tracked_cells_df)):
        
        cell = tracked_cells_df.iloc[idx]
    
        x = cell.X
        y = cell.Y
        z = cell.Z
        
        
        ### ENSURE ORDER OF XY IS CORRECT HERE!!!
        scale = mesh[int(y), int(x)]
    
        new_z = z - scale
        
        ### DEBUG
        #m[int(y), int(x)] = 0
        cell.Z = new_z
    
        tracked_cells_df.iloc[idx] = cell


    """ Set globally """
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    ax_title_size = 18
    leg_size = 18
    plt.rcParams['figure.dpi'] = 300

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
    

    all_norm_tots.append(norm_tots_ALL); all_norm_new.append(norm_new_ALL);
    all_norm_t32.append(norm_tots_32) ; all_norm_n32.append(norm_new_32);
    all_norm_t65.append(norm_tots_65); all_norm_n65.append(norm_new_65);
    all_norm_t99.append(norm_tots_99); all_norm_n99.append(norm_new_99);
    all_norm_t132.append(norm_tots_132); all_norm_n132.append(norm_new_132);
    all_norm_t165.append(norm_tots_165); all_norm_n165.append(norm_new_165);




if not control:
    sav_dir = './OUTPUT_plots_pooled_data_0/'
    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("Directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , sav_dir ,  " already exists")
        
else:
    sav_dir = './OUTPUT_plots_pooled_data_CONTROL/'
    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("Directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , sav_dir ,  " already exists")    
        


""" Also pool things from WITHIN the same experiment """
folder_pools
copy_pooled = np.copy(all_norm_tots)

def pool_lists(all_norm_tots, folder_pools):
    pooled_indices = []; pooled_lists = [];
    for idx in range(1, int(np.max(folder_pools)) + 1, 1):
        
        to_pool = np.where(folder_pools == idx)[0]
        pooled_indices.append(to_pool)
        print(idx)
        
        
        p_arr = all_norm_tots[int(to_pool[0])]
        for p_idx in to_pool[1:len(to_pool)]:
        
            
            add = p_arr[0:len(all_norm_tots[p_idx])] + all_norm_tots[p_idx]
            
            p_arr[0:len(all_norm_tots[p_idx])] = add
            
            
            ### double the indices not pooled
            if len(all_norm_tots[p_idx]) < len(p_arr): 
                
                p_arr[-1] = p_arr[-1] * 2
               
           
        #print(p_arr)
        p_arr = p_arr/len(to_pool)
        
        pooled_lists.append(p_arr)
    
    pooled_indices = np.unique(pooled_indices)
        
    ### then get the unpooled folders:
    for idx, row in enumerate(all_norm_tots):
        
        if idx in pooled_indices:
            continue;
        else:
            pooled_lists.append(row)
            
    return pooled_lists


all_norm_tots = pool_lists(all_norm_tots, folder_pools)
all_norm_new = pool_lists(all_norm_new, folder_pools)
all_norm_t32 = pool_lists(all_norm_t32, folder_pools)
all_norm_n32 = pool_lists(all_norm_n32, folder_pools)


all_norm_t65 = pool_lists(all_norm_t65, folder_pools)
all_norm_n65 = pool_lists(all_norm_n65, folder_pools)

all_norm_t99 = pool_lists(all_norm_t99, folder_pools)
all_norm_n99 = pool_lists(all_norm_n99, folder_pools)
all_norm_t132 = pool_lists(all_norm_t132, folder_pools)
all_norm_n132 = pool_lists(all_norm_n132, folder_pools)

all_norm_t165 = pool_lists(all_norm_t165, folder_pools)
all_norm_n165 = pool_lists(all_norm_n165, folder_pools)



""" If week was skipped over in imaging, then set 0 to np.nan"""
for row in all_norm_tots:    row[row == 0] = np.nan
for row in all_norm_new:    row[row == 0] = np.nan
for row in all_norm_t32:    row[row == 0] = np.nan
for row in all_norm_n32:    row[row == 0] = np.nan
for row in all_norm_t65:    row[row == 0] = np.nan
for row in all_norm_n65:    row[row == 0] = np.nan
for row in all_norm_t99:    row[row == 0] = np.nan
for row in all_norm_n99:    row[row == 0] = np.nan
for row in all_norm_t132:    row[row == 0] = np.nan
for row in all_norm_n132:    row[row == 0] = np.nan
for row in all_norm_t165:    row[row == 0] = np.nan
for row in all_norm_n165:    row[row == 0] = np.nan



lens = []; 
for row in all_norm_n165: lens.append(len(row))

if not control:
    max_week = np.max(lens) - 2  ### stop at week 12
else:
    max_week = np.max(lens)


plot_pooled_trends(all_norm_tots, all_norm_new, ax_title_size, sav_dir, add_name='OUTPUT_overall_', max_week=max_week)
plot_pooled_trends(all_norm_t32, all_norm_n32, ax_title_size, sav_dir, add_name='OUTPUT_overall_0-32', max_week=max_week)
plot_pooled_trends(all_norm_t65, all_norm_n65, ax_title_size, sav_dir, add_name='OUTPUT_overall_32-65', max_week=max_week)
plot_pooled_trends(all_norm_t99, all_norm_n99, ax_title_size, sav_dir, add_name='OUTPUT_overall_65_99', max_week=max_week)
plot_pooled_trends(all_norm_t132, all_norm_n132, ax_title_size, sav_dir, add_name='OUTPUT_overall_99-132', max_week=max_week)
plot_pooled_trends(all_norm_t165, all_norm_n165, ax_title_size, sav_dir, add_name='OUTPUT_overall_132-165', max_week=max_week)



""" Pool all tracked_cells_df and add np.max() so series numbers don't overlap! """
if control:
    #scales = [5.5, 10.6, 9.3, 10.64, 8.7, 11.6]
    scales = [6.74, 11.12, 9.43, 11.12, 9, 11.83]
    
    
    scales = [6.311, 9.6, 8.87, 7.59, 10.9]
    
else:
    #scales = [11.12, 13.92, 11.42, 7.41, 17.6, 17.25]
    scales = [10, 13.67, 12.23, 7.9, 15.16, 15.53]
    
    scales = [11.25, 13.7, 12.3, 7.76, 16.36, 15.46]
    
    
    
    

""" Pool all tracked_cells_df and add np.max() so series numbers don't overlap! """
pooled_tracks = all_tracked_cells_df[0]
pooled_tracks['scale'] = scales[0]
for idx, tracks in enumerate(all_tracked_cells_df):
    
    max_series = np.max(pooled_tracks.SERIES)    
    
    if idx > 0:
        tracks.SERIES = tracks.SERIES + max_series    
        
        tracks['scale'] = scales[idx]
        
        pooled_tracks = pd.concat([pooled_tracks, tracks])
    
    
tracked_cells_df = pooled_tracks

""" Save pooled: """
if not control:
    tracked_cells_df.to_pickle(sav_dir + 'tracked_cells_df_POOLED_CLEANED.pkl')
else:
    tracked_cells_df.to_pickle(sav_dir + 'tracked_cells_df_CONTROL_POOLED.pkl')


if control:
    zzz


""" 
    Also do density analysis of where new cells pop-up???

"""        
analyze = 1;    
if analyze == 1:
    neighbors = 10
    

    ### ***MUST BE RUN LOADING IN FOLDER WITH LONGEST TIMESERIES FIRST!!! OTHERWISE WILL NOT COMPILE!!!
    for idx, tracked in enumerate(all_tracked_cells_df):

        new_cells_per_frame = [[] for _ in range(np.max(tracked.FRAME) + 1)]
        terminated_cells_per_frame = [[] for _ in range(np.max(tracked.FRAME) + 1)]
 
 
 
        for cell_num in np.unique(tracked.SERIES):
        
            frames_cur_cell = tracked.iloc[np.where(tracked.SERIES == cell_num)].FRAME
            
            beginning_frame = np.min(frames_cur_cell)
            if beginning_frame > 0:   # skip the first frame
                new_cells_per_frame[beginning_frame].append(cell_num)
                        
            len_frames = len(np.unique(tracked.FRAME))
            
            
            term_frame = np.max(frames_cur_cell)
            if term_frame < len_frames - 1:   # skip the last frame
                terminated_cells_per_frame[term_frame].append(cell_num)
        


        ### loop through each frame and all the new cells and find "i.... i + n" nearest neighbors        
        """ Plt density of NEW cells vs. depth """
        scaled_vol = 1
        total_dists, total_vols, total_z, new_dists, term_dists, new_vol, term_vol, new_z, term_z, new_EXCLUDE, new_EXCLUDE_z, term_EXCLUDE, term_EXCLUDE_z = plot_density_and_volume(tracked, new_cells_per_frame, terminated_cells_per_frame, scale_xy, scale_z, sav_dir, neighbors, ax_title_size, leg_size, scaled_vol=scaled_vol, plot=0)



        ### initialize if first time
        if idx == 0:
            all_total_dists = total_dists
            all_total_vols = total_vols
            all_total_z = total_z
            all_new_dists = new_dists
            all_term_dists = term_dists
            all_new_vol = new_vol
            all_term_vol = term_vol
            all_new_z = new_z
            all_term_z = term_z
            
            
            all_new_EXCLUDE = new_EXCLUDE
            all_new_EXCLUDE_z = new_EXCLUDE_z
            
            all_term_EXCLUDE = term_EXCLUDE
            all_term_EXCLUDE_z = term_EXCLUDE_z
            
            
        else:  ### else concatenate new values
            for idx, row in enumerate(total_dists): all_total_dists[idx] = np.concatenate((all_total_dists[idx], row))
            for idx, row in enumerate(total_vols): all_total_vols[idx] = np.concatenate((all_total_vols[idx], row))
            for idx, row in enumerate(total_z): all_total_z[idx] = np.concatenate((all_total_z[idx], row))
            for idx, row in enumerate(new_dists): all_new_dists[idx] = np.concatenate((all_new_dists[idx], row))
            for idx, row in enumerate(term_dists): all_term_dists[idx] = np.concatenate((all_term_dists[idx], row))
            for idx, row in enumerate(new_vol): all_new_vol[idx] = np.concatenate((all_new_vol[idx], row))
            for idx, row in enumerate(term_vol): all_term_vol[idx] = np.concatenate((all_term_vol[idx], row))
            for idx, row in enumerate(new_z): all_new_z[idx] = np.concatenate((all_new_z[idx], row))
            for idx, row in enumerate(term_z): all_term_z[idx] = np.concatenate((all_term_z[idx], row))


            for idx, row in enumerate(new_EXCLUDE): all_new_EXCLUDE[idx] = np.concatenate((all_new_EXCLUDE[idx], row))
            for idx, row in enumerate(new_EXCLUDE_z): all_new_EXCLUDE_z[idx] = np.concatenate((all_new_EXCLUDE_z[idx], row))
            for idx, row in enumerate(term_EXCLUDE): all_term_EXCLUDE[idx] = np.concatenate((all_term_EXCLUDE[idx], row))
            for idx, row in enumerate(term_EXCLUDE_z): all_term_EXCLUDE_z[idx] = np.concatenate((all_term_EXCLUDE_z[idx], row))
            
            print('append')
   
    plot_DENSITY_VOLUME_GRAPHS(all_total_dists, all_total_vols, all_total_z, all_new_dists, all_term_dists, all_new_vol, all_term_vol, all_new_z, all_term_z, sav_dir, neighbors, ax_title_size, leg_size, name = '', figsize=(6,5))         
   
    
    """ 
            LOAD IN CONTROL DATA: 
        """
    if not control:
        load_dir = './OUTPUT_plots_pooled_data_CONTROL/'
        tracked_cells_CONTROL = pd.read_pickle(load_dir + 'tracked_cells_df_CONTROL_POOLED.pkl')

        """ get new cells from control """
        mean_control_new, sem_control_new, tracks_control_new, z_control_new, list_indices = plot_size_decay_in_recovery(tracked_cells_CONTROL, sav_dir, start_end_NEW_cell=[1, len(np.unique(tracked_cells_CONTROL.FRAME))],
                                                                                       min_survive_frames=3, use_scaled=1, y_lim=8000, last_plot_week=len(np.unique(tracked_cells_CONTROL.FRAME)),
                                                                                       ax_title_size=ax_title_size, x_label='Weeks no treatment', intial_week=0, figsize=(6, 5))

        """ After cleaning, these are the rows to delete: """
        #zzz

        to_del = [1,2,3,4,5,6,8,10,13,15,27,31,44,56,58,59,60,64,69,70]
        tracks_control_new = np.delete(tracks_control_new, to_del, axis=0)
        mean_control_new = np.nanmedian(tracks_control_new, axis=0)
        std = np.nanstd(tracks_control_new, axis=0)

        ### find sem
        all_n = []
        for col_idx in range(len(tracks_control_new[0, :])):
            all_n.append(len(np.where(~np.isnan(tracks_control_new[:,col_idx]))[0]))
            
        sem_control_new = std/np.sqrt(all_n)
                    

    """ 
        Plot size decay for each frame STARTING from recovery
        
        
            ***add way to choose what frame to stop/start on
            - return the averages
            - change name to whatever is appropriate
    """     
    plt.close('all');
    if not control:
        
        """ Only plot those below 300 um """
        tracked_below_100 = tracked_cells_df.iloc[np.where(tracked_cells_df.Z < 100)[0]]
        
        mean_recovery, sem_recovery, tracks_recovery, z_recovery, list_indices = plot_size_decay_in_recovery(tracked_below_100, sav_dir, start_end_NEW_cell=[4, 8],
                                                                                   min_survive_frames=3, use_scaled=1, y_lim=8000, last_plot_week=len(np.unique(tracked_cells_CONTROL.FRAME)),
                                                                                   ax_title_size=ax_title_size, x_label='Weeks of recovery', figsize=(7, 5))


        """ Also plot for cells that were just on very first frame """
        mean_recovery_short, sem_recovery_short, tracks_recovery_short, z_recovery_short, list_indices = plot_size_decay_in_recovery(tracked_below_100, sav_dir, start_end_NEW_cell=[0, 0],
                                                                           min_survive_frames=3, use_scaled=1, y_lim=8000, last_plot_week=len(np.unique(tracked_cells_CONTROL.FRAME)),
                                                                           ax_title_size=ax_title_size, x_label='Weeks of recovery continued', figsize=(7, 5))
        

        """ Also plot for cells that make it PAST cuprizone and keep going """
        mean_recovery_long, sem_recovery_long, tracks_recovery_long, z_recovery_long, list_indices = plot_size_decay_in_recovery(tracked_below_100, sav_dir, start_end_NEW_cell=[0, 0],
                                                                           min_survive_frames=12, use_scaled=1, y_lim=8000, last_plot_week=len(np.unique(tracked_cells_CONTROL.FRAME)),
                                                                           ax_title_size=ax_title_size, x_label='Weeks of recovery continued', figsize=(7, 5))
        
        """ get decay constant of recovery size change """
        # from scipy.optimize import curve_fit
        # # define type of function to search
        # def model_func(x, a, b, c):
        #     return a*np.exp(-b*x)+c

        # x = np.asarray(np.arange(len(mean_recovery)))
        # y = mean_recovery        
       
        # # curve fit
        # p0 = (1.,1.e-5,1.) # starting search koefs
        
        # p0=[500,10,0]
        
        # opt, pcov = curve_fit(model_func, x, y, p0, maxfev=1000)
        # a, k, b = opt
        # # test result
        # x2 = np.linspace(0, 8, 100)
        # y2 = model_func(x2, a, k, b)
        # fig, ax = plt.subplots()
        # ax.plot(x2, y2, color='r', label='Fit. func: $f(x) = %.3f e^{%.3f x} %+.3f$' % (a,k,b))
        # ax.plot(x, y, 'bo', label='data with noise')
        # ax.legend(loc='best')
        # plt.show()              
        
        """ More imports """
        import statsmodels.api as sa
        import scikit_posthocs as sp
        
        
        """ Anova stats """
        import scipy.stats as scipy
        import statsmodels.api as sm
        import statsmodels.stats.multicomp
        import scipy.stats as stats
        
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
                
        
        all_tracks = np.vstack(tracks_recovery)
        all_tracks_df = pd.DataFrame(data=all_tracks)
        all_tracks_df = all_tracks_df.dropna()
        
        
        F, p = stats.f_oneway(all_tracks_df[0],all_tracks_df[1],all_tracks_df[2],all_tracks_df[3])
        # Seeing if the overall model is significant
        print('F-Statistic=%.3f, p=%.3f' % (F, p)) 
        
        
        stacked_data = all_tracks_df.stack().reset_index()
        stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                                    'level_1': 'week',
                                                    0:'size'})

        
        mc = statsmodels.stats.multicomp.MultiComparison(stacked_data['size'],stacked_data['week'])
        mc_results = mc.tukeyhsd()
        #print(mc_results)      

        """ Do with Kruskal wallis"""
        #print(sp.posthoc_dunn(stacked_data, val_col='size', group_col='week'))
        #p_vals_kruskal = sp.posthoc_mannwhitney(stacked_data, val_col='size', group_col='week')
        
        p_vals_kruskal = sp.posthoc_dunn(stacked_data, val_col='size', group_col='week')
        for p in p_vals_kruskal.iterrows():
            print(round(p[1], 5))
        
        
        
        # from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
        # from statsmodels.stats.libqsturng import psturng
                
        # p_values = psturng(np.abs(mc_results.meandiffs / mc_results.std_pairs), len(mc_results.groupsunique), mc_results.df_total)
        # print(p_values)
        
        
        
        """ Do for cuprizone """
                        
                
        mean_cuprizone, sem_cuprizone, tracks_cuprizone, z_cuprizone, list_indices = plot_size_decay_in_recovery(tracked_below_100, sav_dir, start_end_NEW_cell=[0, 0],
                                                                                   min_survive_frames=0, use_scaled=1, y_lim=8000, ax_title_size=ax_title_size,
                                                                                   x_label='Weeks of cuprizone', intial_week=0, last_plot_week=4, figsize=(6, 5))        
        
        """ Anova stats """
        all_tracks = np.vstack(tracks_cuprizone)
        all_tracks_df = pd.DataFrame(data=all_tracks)
        all_tracks_df = all_tracks_df.dropna()
        
        
        F, p = stats.f_oneway(all_tracks_df[0],all_tracks_df[1],all_tracks_df[2],all_tracks_df[3])
        # Seeing if the overall model is significant
        print('F-Statistic=%.3f, p=%.3f' % (F, p)) 
        
        
        stacked_data = all_tracks_df.stack().reset_index()
        stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                                    'level_1': 'week',
                                                    0:'size'})

        
        mc = statsmodels.stats.multicomp.MultiComparison(stacked_data['size'],stacked_data['week'])
        mc_results = mc.tukeyhsd()
        #print(mc_results)
        
        
        """ Do with Kruskal wallis"""
        #print(sp.posthoc_dunn(stacked_data, val_col='size', group_col='week'))
        #p_vals_kruskal = sp.posthoc_mannwhitney(stacked_data, val_col='size', group_col='week')
        
        p_vals_kruskal = sp.posthoc_dunn(stacked_data, val_col='size', group_col='week')
        for p in p_vals_kruskal.iterrows():
            print(round(p[1], 5))
        

               

    """ get control values """
    mean_control, sem_control, tracks_control, tracks_control_unscaled, list_indices = plot_size_decay_in_recovery(tracked_cells_CONTROL, sav_dir, start_end_NEW_cell=[0, 0],
                                                                                   min_survive_frames=3, use_scaled=1, y_lim=8000, last_plot_week=len(np.unique(tracked_cells_CONTROL.FRAME)),
                                                                                   ax_title_size=ax_title_size, x_label='Weeks no treatment', intial_week=0, figsize=(6, 5))



    """ Anova stats """
    all_tracks = np.vstack(tracks_control)
    all_tracks_df = pd.DataFrame(data=all_tracks)
    all_tracks_df = all_tracks_df.dropna()
    
    
    F, p = stats.f_oneway(all_tracks_df[0],all_tracks_df[1],all_tracks_df[2],all_tracks_df[3])
    # Seeing if the overall model is significant
    print('F-Statistic=%.3f, p=%.3f' % (F, p)) 
    
    
    stacked_data = all_tracks_df.stack().reset_index()
    stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                                'level_1': 'week',
                                                0:'size'})

    
    mc = statsmodels.stats.multicomp.MultiComparison(stacked_data['size'],stacked_data['week'])
    
    mc_results = mc.tukeyhsd()
    
    ### WITH BONFERRI correction
    # mc_results = mc.allpairtest(stats.ttest_rel, method='Holm')
    
    
    #print(mc_results[0])        
                        

    """ Plot: """
    ### (1) control vs. recovery, starting at week 4
    mean_c = mean_control[1:5]
    sem_c = sem_control[1:5]
    tracks_c = tracks_control[:, 1:5]
    
    mean_r = mean_recovery[0:len(mean_c)]
    sem_r = sem_recovery[0:len(sem_c)]
    tracks_r = tracks_recovery[:, 0:len(sem_c)]


    """ Normalize more """
    tracks_r_norm = tracks_r/mean_c
    
    """ NORMALIZE TO CONTROL """
    mean = np.nanmedian(tracks_r_norm, axis=0)
    std = np.nanstd(tracks_r_norm, axis=0)

    ### find sem
    all_n = []
    for col_idx in range(len(tracks_r_norm[0, :])):
        all_n.append(len(np.where(~np.isnan(tracks_r_norm[:,col_idx]))[0]))
        
    sem = std/np.sqrt(all_n)
    
    mean_r = mean
    sem_r = sem


    


    
    """ Normalize to control """
    ### also get new cells in CONTROL
    tracks_c_NEW = tracks_control_new[:, :4]
    tracks_c_NEW = tracks_c_NEW/mean_c
    
    mean = np.nanmedian(tracks_c_NEW, axis=0)
    std = np.nanstd(tracks_c_NEW, axis=0)

    ### find sem
    all_n = []
    for col_idx in range(len(tracks_c_NEW[0, :])):
        all_n.append(len(np.where(~np.isnan(tracks_c_NEW[:,col_idx]))[0]))
        
    sem = std/np.sqrt(all_n)
    
    mean_control_new = mean
    sem_control_new = sem    
    
    
    


    """ Setup 2-way ANOVA for 
    
            Size ==> dependent variable
            Week & treatment ==> independent variable
    """
    tracks_c = tracks_c/mean_c
    
    ### setup control
    all_tracks = np.vstack(tracks_c)
    all_tracks_df = pd.DataFrame(data=all_tracks)
    all_tracks_df = all_tracks_df.dropna()        
    
    stacked_data = all_tracks_df.stack().reset_index()
    stacked_data_control = stacked_data.rename(columns={'level_0': 'condition',
                                                'level_1': 'week',
                                                0:'size'})
    stacked_data_control.condition = 0
    
    
    ### setup recovery
    all_tracks = np.vstack(tracks_r_norm)
    all_tracks_df = pd.DataFrame(data=all_tracks)
    all_tracks_df = all_tracks_df.dropna()        
    
    stacked_data = all_tracks_df.stack().reset_index()
    stacked_data_rec = stacked_data.rename(columns={'level_0': 'condition',
                                                'level_1': 'week',
                                                0:'size'})
    stacked_data_rec.condition = 1
    
    
    
    ### setup control NEW cells
    
    all_tracks = np.vstack(tracks_c_NEW)
    all_tracks_df = pd.DataFrame(data=all_tracks)
    all_tracks_df = all_tracks_df.dropna()        
    
    stacked_data = all_tracks_df.stack().reset_index()
    stacked_data_control_NEW = stacked_data.rename(columns={'level_0': 'condition',
                                                'level_1': 'week',
                                                0:'size'})
    stacked_data_control_NEW.condition = 2
    
    
    
    
    concat_df = pd.concat([stacked_data_control, stacked_data_rec, stacked_data_control_NEW])
    
    # mc = statsmodels.stats.multicomp.MultiComparison(concat_df['size'], np.asarray(concat_df[['condition', 'week']]))
    # mc_results = mc.tukeyhsd()
    # print(mc_results)        
                            

    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    for name, grouped_df in concat_df.groupby('week'):
        #print('Name {}'.format(name), pairwise_tukeyhsd(grouped_df['size'], grouped_df['condition']))

        #print(sp.posthoc_mannwhitney(grouped_df, val_col='size', group_col='condition'))

        """ Do with Kruskal wallis"""
        #print(sp.posthoc_dunn(grouped_df, val_col='size', group_col='condition'))
        #p_vals_kruskal = sp.posthoc_mannwhitney(stacked_data, val_col='size', group_col='week')
        
        p_vals_kruskal = sp.posthoc_dunn(grouped_df, val_col='size', group_col='condition')
        print('\nWeek split')
        for p in p_vals_kruskal.iterrows():
            
            print(round(p[1], 5))

        """ Effect size """
        group_one = grouped_df.iloc[np.where((grouped_df['condition'] == 0))]['size']
        group_two = grouped_df.iloc[np.where((grouped_df['condition'] == 1))]['size']
        group_three = grouped_df.iloc[np.where((grouped_df['condition'] == 2))]['size']
        
        effect_size = cohen_d(group_one, group_two)
        print('effect size for baseline vs. RECOVERY for size: ' + str(effect_size))

        effect_size = cohen_d(group_one, group_three)
        print('effect size for baseline vs. CONTROL_NEW for size: ' + str(effect_size))



    # concat_df = pd.concat([stacked_data_control, stacked_data_control_NEW])
    # from statsmodels.stats.multicomp import pairwise_tukeyhsd
    # for name, grouped_df in concat_df.groupby('week'):
    #     print('Name {}'.format(name), pairwise_tukeyhsd(grouped_df['size'], grouped_df['condition']))
        
        
    

    """ continue with plotting """
    
    plt.figure(figsize=(5, 5));
    # plt.plot(np.arange(1, len(mean_c) + 1), mean_c, color='k', linestyle='dotted', linewidth=2)
    # plt.scatter(np.arange(1, len(mean_c) + 1), mean_c, color='k', marker='o', s=30)
    # plt.errorbar(np.arange(1, len(mean_c) + 1), mean_c, yerr=sem_c, color='k', fmt='none', capsize=0, capthick=0)


    plt.plot(np.arange(1, len(mean_c) + 1), np.ones(len(mean_c)), color='k', linestyle='dotted', linewidth=2)
    plt.scatter(np.arange(1, len(mean_c) + 1), np.ones(len(mean_c)), color='k', marker='o', s=30)
    #plt.errorbar(np.arange(1, len(mean_c) + 1), np.ones(len(mean_c)), yerr=sem_c, color='k', fmt='none', capsize=0, capthick=0)
    
    
    
    plt.plot(np.arange(1, len(mean_r) + 1), mean_r, color='r', linestyle='dotted', linewidth=2)
    plt.scatter(np.arange(1, len(mean_r) + 1), mean_r, color='r', marker='o', s=30)
    plt.errorbar(np.arange(1, len(mean_r) + 1), mean_r, yerr=sem_r, color='r', fmt='none', capsize=0, capthick=0)    
    

    ### ALSO PLOT NEW CELLS IN CONTROL
    plt.plot(np.arange(1, len(mean_control_new) + 1), mean_control_new, color='royalblue', linestyle='dotted', linewidth=2)
    plt.scatter(np.arange(1, len(mean_control_new) + 1), mean_control_new, color='royalblue', marker='o', s=30)
    plt.errorbar(np.arange(1, len(mean_control_new) + 1), mean_control_new, yerr=sem_control_new, color='royalblue', fmt='none', capsize=0, capthick=0)    
    

    plt.ylim(0, 2)
    
    plt.xlabel('Weeks', fontsize=ax_title_size)
    plt.ylabel('Normalized soma volume', fontsize=ax_title_size)
    ax = plt.gca()
    
    from matplotlib.ticker import MaxNLocator   ### force integer tick sizes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    rs = ax.spines["right"]; rs.set_visible(False)
    ts = ax.spines["top"]; ts.set_visible(False)  
    plt.tight_layout()
    plt.legend(['Control stable', 'Recovery', 'Control new'], frameon=False, fontsize=leg_size, loc='lower right')
    plt.savefig(sav_dir + '_COMPARE recovery to control curves SIZE.png')



    ### (2) control vs. cuprizone, starting at week 4
    mean_c = mean_control[1:5]
    sem_c = sem_control[1:5]
    tracks_c = tracks_control[:, 1:5]

    
    mean_cup = mean_cuprizone[0:4]
    sem_cup = sem_cuprizone[0:4]
    tracks_cup = tracks_cuprizone[:, 0:4]
    
    
    """ Normalize to control """
    ### also get new cells in CONTROL
    tracks_cup = tracks_cup/mean_c
    
    mean = np.nanmedian(tracks_cup, axis=0)
    std = np.nanstd(tracks_cup, axis=0)

    ### find sem
    all_n = []
    for col_idx in range(len(tracks_cup[0, :])):
        all_n.append(len(np.where(~np.isnan(tracks_cup[:,col_idx]))[0]))
        
    sem = std/np.sqrt(all_n)
    
    mean_cup = mean
    sem_cup = sem    
    
    
    

    plt.figure(figsize=(5, 5));
    # plt.plot(np.arange(0, len(mean_c)), mean_c, color='k', linestyle='dotted', linewidth=2)
    # plt.scatter(np.arange(0, len(mean_c)), mean_c, color='k', marker='o', s=30)
    # plt.errorbar(np.arange(0, len(mean_c)), mean_c, yerr=sem_c, color='k', fmt='none', capsize=0, capthick=0)
    
    
    plt.plot(np.arange(0, len(mean_c)), np.ones(len(mean_c)), color='k', linestyle='dotted', linewidth=2)
    plt.scatter(np.arange(0, len(mean_c)), np.ones(len(mean_c)), color='k', marker='o', s=30)
    #plt.errorbar(np.arange(1, len(mean_c) + 1), np.ones(len(mean_c)), yerr=sem_c, color='k', fmt='none', capsize=0, capthick=0)
        
    
    plt.plot(np.arange(0, len(mean_cup)), mean_cup, color='r', linestyle='dotted', linewidth=2)
    plt.scatter(np.arange(0, len(mean_cup)), mean_cup, color='r', marker='o', s=30)
    plt.errorbar(np.arange(0, len(mean_cup)), mean_cup, yerr=sem_cup, color='r', fmt='none', capsize=0, capthick=0)    
    

    plt.ylim(0, 2)
    
    plt.xlabel('Weeks', fontsize=ax_title_size)
    plt.ylabel('Normalized soma volume', fontsize=ax_title_size)
    ax = plt.gca()
    
    from matplotlib.ticker import MaxNLocator   ### force integer tick sizes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    rs = ax.spines["right"]; rs.set_visible(False)
    ts = ax.spines["top"]; ts.set_visible(False)  
    plt.tight_layout()
    plt.legend(['Control', 'Cuprizone'], frameon=False, fontsize=leg_size)
    plt.savefig(sav_dir + '_COMPARE cuprizone to control curves SIZE.png')



    """ Setup 2-way ANOVA for 
            Size ==> dependent variable
            Week & treatment ==> independent variable
    """
    tracks_c = tracks_c/mean_c
    all_tracks = np.vstack(tracks_c)
    all_tracks_df = pd.DataFrame(data=all_tracks)
    all_tracks_df = all_tracks_df.dropna()        
    
    stacked_data = all_tracks_df.stack().reset_index()
    stacked_data_control = stacked_data.rename(columns={'level_0': 'condition',
                                                'level_1': 'week',
                                                0:'size'})
    stacked_data_control.condition = 0
    
    
    all_tracks = np.vstack(tracks_cup)
    all_tracks_df = pd.DataFrame(data=all_tracks)
    all_tracks_df = all_tracks_df.dropna()        
    
    stacked_data = all_tracks_df.stack().reset_index()
    stacked_data_rec = stacked_data.rename(columns={'level_0': 'condition',
                                                'level_1': 'week',
                                                0:'size'})
    stacked_data_rec.condition = 1
    
    
    concat_df = pd.concat([stacked_data_control, stacked_data_rec])
    
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    for name, grouped_df in concat_df.groupby('week'):
        #print('Name {}'.format(name), pairwise_tukeyhsd(grouped_df['size'], grouped_df['condition']))
        
        
        #print(sp.posthoc_conover(grouped_df, val_col='size', group_col='condition', p_adjust='holm'))
        
        
        """ Do with Kruskal wallis"""
        #print(sp.posthoc_dunn(grouped_df, val_col='size', group_col='condition'))
        #p_vals_kruskal = sp.posthoc_mannwhitney(stacked_data, val_col='size', group_col='week')
        
        p_vals_kruskal = sp.posthoc_dunn(grouped_df, val_col='size', group_col='condition')
        print('\nWeek split')
        for p in p_vals_kruskal.iterrows():
            print(round(p[1], 5))
 
        """ Effect size """
        group_one = grouped_df.iloc[np.where((grouped_df['condition'] == 0))]['size']
        group_two = grouped_df.iloc[np.where((grouped_df['condition'] == 1))]['size']
        
        effect_size = cohen_d(group_one, group_two)
        print('effect size for baseline vs. combined for size: ' + str(effect_size))


        # t_test = scipy.stats.ttest_ind(group_one, group_two, nan_policy='omit')
        # print('p-value for baseline vs. combined for size: ' + str(t_test.pvalue))

                
 


    """ Plot scatters of each type:
            - control/baseline day 1 ==> frame 0
                    ***cuprizone ==> frame 4
            - 1 week after cupr
            - 2 weeks after cupr
            - 3 weeks after cupr 
        
        """
    if not control:
        first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week = plot_size_scatters_by_recovery(tracked_below_100, sav_dir, start_frame=4, end_frame=8, 
                                                                                                                   min_survive_frames=3, use_scaled=1, y_lim=10000, ax_title_size=ax_title_size)
   
        ### compare to control cells pooled from the same timepoints
        tracks_c = tracks_control_unscaled[:, 4:]
        tracks_c = tracks_c[~np.isnan(tracks_c)]
        print('Number of control cells: ' + str(len(tracks_c)))
        first_frame_sizes = np.transpose(tracks_c.flatten())
    
        data = {'Control':first_frame_sizes, '1 week\nold cells':first_frame_1_week, '2 week\nold cells': first_frame_2_week, '3 week\nold cells': first_frame_3_week}
        #df = pd.DataFrame(data=data)
        
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))
        
        
        plt.figure(figsize=(6.25,5))
        # Make a dictionary with one specific color per group:
      
        x = sns.violinplot(data=df, palette=[(0.4, 0.4, 0.4), (0, 1.0, 0),(0, 0.7, 0),(0, 0.4, 0)])
        plt.ylim(-1500, 10000)   ### leave room for significance!!!
        
        plt.ylabel('Soma volume ($\u03bcm^3$)', fontsize=ax_title_size)
        ax = plt.gca()
        rs = ax.spines["right"]; rs.set_visible(False)
        ts = ax.spines["top"]; ts.set_visible(False)  
        ax.grid(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='black')
        plt.tight_layout()
        plt.savefig(sav_dir + 'Cell sizes changing.png')
           
        
        """ ANOVA """
        df = df.dropna()
        F, p = stats.f_oneway(df['Control'],df['1 week\nold cells'],df['2 week\nold cells'],df['3 week\nold cells'])
        # Seeing if the overall model is significant
        print('F-Statistic=%.3f, p=%.3f' % (F, p)) 
        
        
        stacked_data = df.stack().reset_index()
        stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                                    'level_1': 'week',
                                                    0:'size'})
    
        mc = statsmodels.stats.multicomp.MultiComparison(stacked_data['size'],stacked_data['week'])
        mc_results = mc.tukeyhsd()
        print(mc_results)    


        # from statsmodels.stats.multicomp import pairwise_tukeyhsd
        # for name, grouped_df in concat_df.groupby('week'):
        #     #print('Name {}'.format(name), pairwise_tukeyhsd(grouped_df['size'], grouped_df['condition']))
            
            
        #     #print(sp.posthoc_conover(grouped_df, val_col='size', group_col='condition', p_adjust='holm'))
            
            
        #     """ Do with Kruskal wallis"""
        #     #print(sp.posthoc_dunn(grouped_df, val_col='size', group_col='condition'))
        #     #p_vals_kruskal = sp.posthoc_mannwhitney(stacked_data, val_col='size', group_col='week')
            
        #     p_vals_kruskal = sp.posthoc_dunn(grouped_df, val_col='size', group_col='week')
        #     print('\nWeek split')
        #     for p in p_vals_kruskal.iterrows():
        #         print(round(p[1], 5))

    """ Predict age based on size??? 
    
            what is probability that cell is 1 week old P(B) given that it is size X P(A) == P(B|A) == P(A and B) / P(A)
            P(A) == prob cell is ABOVE size X
            P(B) == prob cell 1 week old
            P(A and B) == prob cell is at least 1 week old AND above size X
            
            
            
            ***PROBABILITY compared to time-point paired controls!!!
                i.e. get control cells from 1 week, 2 weeks and 3 weeks after cuprizone treatment
            
            
            
            
    """
    """ DOUBLE CHECK THIS PROBABILITY CALCULATION!!!"""
    
    upper_r = 6000
    lower_r = 0
    step = 50
    thresh_range = [lower_r, upper_r, step]
    
    
    """    ### SHOULD BE COMPARED TO CELLS ON LATER TIMEFRAMES???? IN TERMS OF SIZE???
    ### because baseline may have some new cells as well!!!
    
                ***compared with cells/themselves that have been around for > 8 weeks???
                
                ****RETROSPECTIVELY CAN SEE IF NEW CELLS ARE INJURED FASTER AT EARLIER TIMEPOINTS by cuprizone???
                ***plot size vs. death time
                    ==> example in 030!!!
                
    """    
    probability_curves(sav_dir, first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week, thresh_range, ax_title_size, leg_size, figsize=(7, 5))

    
    """ Plot density KDE """

    
    import seaborn as sns    
    plt.figure(figsize=(7,5))
    #plt.figure()
    ax = plt.gca()
    sns.set_style('whitegrid')
    rs = ax.spines["right"]; rs.set_visible(False)
    ts = ax.spines["top"]; ts.set_visible(False)  
    

    sns.distplot(np.asarray(first_frame_1_week)/1000, bins=20, hist=False, kde=True, 
                  label='1 week old',axlabel='Soma volume ($mm^3$)', color=(0, 0.9, 0), kde_kws={'linestyle':':', 'linewidth':3})
    sns.distplot(np.asarray(first_frame_2_week)/1000, bins=20, hist=False, kde=True,
                  label='2 weeks old',axlabel='Soma volume ($mm^3$)', color=(0, 0.7, 0), kde_kws={'linestyle':'-.', 'linewidth':3})
    sns.distplot(np.asarray(first_frame_3_week)/1000, bins=20, hist=False, kde=True, 
                  label='3 weeks old', axlabel='Soma volume ($mm^3$)', color=(0, 0.4, 0), kde_kws={'linestyle':'--', 'linewidth':3})
    sns.distplot(np.asarray(first_frame_sizes)/1000, bins=20, hist=False, kde=True, 
                  label='Control', axlabel='Soma volume ($mm^3$)', color=(0, 0, 0), kde_kws={'linestyle':'-', 'linewidth':3})      

    plt.xlabel('Soma volume ($mm^3$)', fontsize=ax_title_size)
    plt.ylabel('Density', fontsize=ax_title_size)

    plt.legend(frameon=False, fontsize=leg_size)
    plt.tight_layout()
    # Turns off grid on the left Axis.
    ax.grid(False)
    ax.spines['bottom'].set_color('black')
    # ax.spines['top'].set_color('#dddddd') 
    # ax.spines['right'].set_color('red')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    
        
    plt.xticks(np.arange(0 , 9))
    plt.xlim(0, 8000/1000 + 0.2)
    plt.ylim(0, 1)
    #plt.xlim(-1, len(y_pos))   
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.savefig(sav_dir + 'KDE density distributions RECOVERY.png')

    """ Plot scatter for dying cells
    
        and probability??? """

    """ Must first get sizes of cell that die on frame 1, frame 2, frame 3"""
    

    
    ### NEED 1 more frame to get row 4 deaths???
    mean_cuprizone, sem_cuprizone, tracks_cuprizone, unscaled_cuprizone, list_indices = plot_size_decay_in_recovery(tracked_below_100, sav_dir, start_end_NEW_cell=[0, 0],
                                                                               min_survive_frames=0, use_scaled=1, y_lim=8000, ax_title_size=ax_title_size,
                                                                               x_label='EXTRA WEEK CUPRIZONE', intial_week=0, last_plot_week=5, figsize=(6, 5))        
    tracks_cup = unscaled_cuprizone        

    died_frame_1 = [];
    died_frame_2 = [];
    died_frame_3 = [];
    died_frame_4 = [];    
    for row in tracks_cup:
        if np.isnan(row[1]):
            died_frame_1.append(row[0])
        elif np.isnan(row[2]):
            died_frame_2.append(row[1])
        elif np.isnan(row[3]):
            died_frame_3.append(row[2])    
        elif np.isnan(row[4]):
            died_frame_4.append(row[3])


    if not control:
        # first_frame_sizes, first_frame_1_week, first_frame_2_week, first_frame_3_week = plot_size_scatters_by_recovery(tracked_cells_df, sav_dir, start_frame=0, end_frame=1, 
        #                                                                                                            min_survive_frames=0, use_scaled=1, y_lim=10000, ax_title_size=ax_title_size)
   
        ### compare to control cells pooled from the same timepoints
        tracks_c = tracks_control_unscaled[:, 0:4]
        tracks_c = tracks_c[~np.isnan(tracks_c)]
        print('Number of control cells: ' + str(len(tracks_c)))
        first_frame_sizes = np.transpose(tracks_c.flatten())
    
        #data = {'Non-treated\ncells':first_frame_sizes, '1 week\ncuprizone':died_frame_2, '2 week\ncuprizone': died_frame_3, '3 week\ncuprizone': died_frame_4}
        
        data = {'Control':first_frame_sizes, 'Size before\ndeath':np.concatenate((died_frame_2, died_frame_3, died_frame_4))}

        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))
        
        plt.figure(figsize=(4.5,5))
        

        ax = sns.violinplot(data=df)
        plt.ylim(-1500, 10000)   ### leave room for significance!!!
        
        plt.ylabel('Soma volume ($\u03bcm^3$)', fontsize=ax_title_size)
        rs = ax.spines["right"]; rs.set_visible(False)
        ts = ax.spines["top"]; ts.set_visible(False)  
        plt.tight_layout()
        plt.savefig(sav_dir + 'DYING CELLS size changing.png')
    
    
        """ STATISTICS """
        # ### UNPAIRED 2-tailed, assume equal variances
        # t_test = sp.stats.ttest_ind(first_frame_sizes, died_frame_1, nan_policy='omit')
        # print('p-value for baseline vs. week 1 of recovery for size: ' + str(t_test.pvalue))
        
        # effect_size = cohen_d(first_frame_sizes, died_frame_1)
        # print('effect size for baseline vs. week 3 of recovery for size: ' + str(effect_size))
        
        # t_test = sp.stats.ttest_ind(first_frame_sizes, died_frame_2, nan_policy='omit')
        # print('p-value for baseline vs. week 2 of recovery for size: ' + str(t_test.pvalue))        
    
        # effect_size = cohen_d(first_frame_sizes, died_frame_2)
        # print('effect size for baseline vs. week 3 of recovery for size: ' + str(effect_size))
    
        
        # t_test = sp.stats.ttest_ind(first_frame_sizes, died_frame_3, nan_policy='omit')
        # print('p-value for baseline vs. week 3 of recovery for size: ' + str(t_test.pvalue))
        
        # effect_size = cohen_d(first_frame_sizes, died_frame_3)
        # print('effect size for baseline vs. week 3 of recovery for size: ' + str(effect_size))
         
        
        
        combined = np.concatenate((died_frame_2, died_frame_3, died_frame_4))
        
        t_test = scipy.stats.ttest_ind(first_frame_sizes, combined, nan_policy='omit')
        print('p-value for baseline vs. combined for size: ' + str(t_test.pvalue))
        
        effect_size = cohen_d(first_frame_sizes, combined)
        print('effect size for baseline vs. combined for size: ' + str(effect_size))
         


                


    """ Plot dying on first frame vs. cell size, to see if larger/newer cells die first? """
    
    thresh_range = [100, 2000, 50]
    all_probs = []; all_sizes = []
    #for size_thresh in range(500, 4100, 100):
    cur_frame = np.concatenate((died_frame_2, died_frame_3, died_frame_4))
        
    for size_thresh in range(thresh_range[0], thresh_range[1], thresh_range[2]):   ### FOR RESCALED VOLUMES
        all_cell_sizes = np.concatenate((first_frame_sizes, died_frame_2, died_frame_3, died_frame_4))
        num_A = 0
        total_num = len(all_cell_sizes)
        for sizes in all_cell_sizes:
            
            #print(size_thresh)
            if sizes < size_thresh:
                num_A += 1
                
        if num_A > 0:

            P_A = num_A/total_num;
    
       
            ### find P(A and B) *** NOT equal to P(A) * P(B) because are INDEPENDENT events!!!
            ### so only way to find is to count # that are BOTH 1 week young AND > size thresh / num total
            num_below_thresh_1_week = len(np.where(np.asarray(cur_frame) < size_thresh)[0])
            ### SHOULD BE DIVIDED BY TOTAL # OF first frame cells??? i.e. total number of cells, not occurences
            #len(np.unique(tracked_cells_df.SERIES))
            
            P_A_B = num_below_thresh_1_week/total_num;
            #P_A_B = num_above_thresh_1_week/len(np.unique(tracked_cells_df.SERIES));
            
            PB_A = P_A_B/P_A
            
            #print(PB_A)
            
            all_probs.append(PB_A)
            all_sizes.append(size_thresh)
    
            
            
    plt.figure(figsize=(4.5,5)); plt.plot(all_sizes, all_probs, 'k')
    ax = plt.gca()
    plt.xlabel('Soma volume ($\u03bcm^3$)', fontsize=ax_title_size); plt.ylabel('Probability death next week', fontsize=ax_title_size)
    plt.ylim(0, 1)   ### leave room for significance!!!

    rs = ax.spines["right"]; rs.set_visible(False)
    ts = ax.spines["top"]; ts.set_visible(False)  
    plt.tight_layout()
    # prop_above_1st_frame = 
    plt.savefig(sav_dir + 'DYING CELLS size probability.png')
    
        


    """ Plot probability cells is dying compared to control population 
    
    """


    










    """
        Plot density of ONLY v235 trial
    """    

    """ Scale Z to mesh """
    #m = np.max(input_im, axis=-1)
    # print('scaling cell Z to mesh')
    # for idx in range(len(two_thirty_tracked)):
        
    #     cell = two_thirty_tracked.iloc[idx]
    
    #     x = cell.X
    #     y = cell.Y
    #     z = cell.Z
        
        
    #     ### ENSURE ORDER OF XY IS CORRECT HERE!!!
    #     scale = mesh_235[int(y), int(x)]
    
    #     new_z = z - scale
        
    #     ### DEBUG
    #     #m[int(y), int(x)] = 0
    #     cell.Z = new_z
    
    #     two_thirty_tracked.iloc[idx] = cell

    # neighbors = 10
    
    # new_cells_per_frame = [[] for _ in range(len(np.unique(two_thirty_tracked.FRAME)) + 1)]
    # terminated_cells_per_frame = [[] for _ in range(len(np.unique(two_thirty_tracked.FRAME)) + 1)]
    # for cell_num in np.unique(two_thirty_tracked.SERIES):
        
    #     frames_cur_cell = two_thirty_tracked.iloc[np.where(two_thirty_tracked.SERIES == cell_num)].FRAME
        
    #     beginning_frame = np.min(frames_cur_cell)
    #     if beginning_frame > 0:   # skip the first frame
    #         new_cells_per_frame[beginning_frame].append(cell_num)
                    
    #     term_frame = np.max(frames_cur_cell)
    #     if term_frame < len(terminated_cells_per_frame) - 1:   # skip the last frame
    #         terminated_cells_per_frame[term_frame].append(cell_num)

    # plot_density_and_volume(two_thirty_tracked, new_cells_per_frame, terminated_cells_per_frame, scale_xy, scale_z, sav_dir, neighbors, 
    #                         ax_title_size, leg_size, scaled_vol=0, name='only_235_')

    






    """
        Density (histogram of cells) by depth on frame 0 (baseline)
            using timeseries v235
    """
    input_path = list_folder[0]
    foldername = input_path.split('/')[-2]
    load_dir_hist = input_path + '/' + foldername + '_output_FULL_AUTO_no_next_10_125762_TEST_6/'
    
    tracked_cells_df_FULL = pd.read_pickle(load_dir_hist + 'tracked_cells_df_RAW_pickle.pkl')
    
    
    """ Scale Z to mesh """
    #m = np.max(input_im, axis=-1)
    print('scaling cell Z to mesh')
    for idx in range(len(tracked_cells_df_FULL)):
        
        cell = tracked_cells_df_FULL.iloc[idx]
    
        x = cell.X
        y = cell.Y
        z = cell.Z
        
        
        ### ENSURE ORDER OF XY IS CORRECT HERE!!!
        scale = mesh_235[int(y), int(x)]
    
        new_z = z - scale
        
        ### DEBUG
        #m[int(y), int(x)] = 0
        cell.Z = new_z
    
        tracked_cells_df_FULL.iloc[idx] = cell

    
    plt.figure(figsize = (13, 2))
    #plt.rcParams['figure.dpi'] = 140        
    all_z_frame_0 = tracked_cells_df_FULL.loc[tracked_cells_df_FULL['FRAME'].isin([0])].Z
    
    all_z_scaled = all_z_frame_0 * scale_z
    
    
    ### only get cells below 420 um
    all_z_scaled = [x for x in all_z_scaled if x <= 420]
    
    plt.hist(all_z_scaled, bins=40, color='gray')
    plt.xlabel('Depth (\u03bcm)', fontsize=ax_title_size)
    plt.ylabel('Num cells', fontsize=ax_title_size)
    ax = plt.gca()
    rs = ax.spines["right"]; rs.set_visible(False)
    ts = ax.spines["top"]; ts.set_visible(False)  
    plt.tight_layout()        
    plt.savefig(sav_dir + 'density along depth.png')
    
    # """ Set globally """
    # plt.rc('xtick',labelsize=16)
    # plt.rc('ytick',labelsize=16)
    # ax_title_size = 18
    # leg_size = 14

        
    