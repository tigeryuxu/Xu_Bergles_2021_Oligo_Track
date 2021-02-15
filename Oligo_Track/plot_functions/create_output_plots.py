# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Tiger
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

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True


""" Set globally """
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)


""" Define GPU to use """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""  Network Begins: """
s_path = './(1) Checkpoints_full_auto_no_spatialW/'
#s_path = './(2) Checkpoints_full_auto_spatialW/'


s_path = './(4) Checkpoints_full_auto_no_spatialW_large_TRACKER/'

crop_size = 160
z_size = 32
num_truth_class = 2

lowest_z_depth = 135;
lowest_z_depth = 150;

scale_for_animation = 0.8



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
    images = glob.glob(os.path.join(input_path,'*_single_channel.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,seg=i.replace('_single_channel.tif','_single_channel_segmentation.tif'), truth=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]

    # images = glob.glob(os.path.join(input_path,'*_input_im.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    # images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    # examples = [dict(input=i,seg=i.replace('_input_im.tif','_segmentation.tif'), truth=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]

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
    
    input_im = input_im[0:lowest_z_depth, ...]
    input_im = np.moveaxis(input_im, 0, -1)
    width_tmp, height_tmp, depth_tmp = input_im.shape
    

    """ Get truth from .csv as well """
    truth = 1
    scale = 1
    
    if truth:
         truth_name = 'MOBPF_190627w_5_syglassCorrectedTracks.csv'; scale = 0
         #truth_name = 'MOBPF_190626w_4_syGlassEdited_20200607.csv';  scale = 1  # cuprizone
         #truth_name = 'a1901128-r670_syGlass_20x.csv'
         #truth_name = '680_syGlass_10x.csv'                           
         
         truth_cur_im, truth_array  = gen_truth_from_csv(frame_num=0, input_path=input_path, filename=truth_name, 
                            input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
         truth_output_df = pd.DataFrame(columns = {'SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z'})
         
 
            
    """ Parse the old array and SAVE IT: """

    #print('num_blobs: ' + str(num_blobs))
    #print('duplicates: ' + str(np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'SERIES']))))   ### cell in same location across frames
    #tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'SERIES'], keep=False))[0]]

    
    ### (1) unsure that all of 'RED' or 'YELLOW' are indicated as such
    ### ***should be fine, just turn all "BLANK" into "GREEN"  
    # tracked_cells_df.COLOR[tracked_cells_df['COLOR'] == 'BLANK'] = 'GREEN'
    
    # num_YELLOW = 0; num_RED = 0 
    # for cell_num in np.unique(tracked_cells_df.SERIES):
       
    #     color_arr = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].COLOR)
        
    #     if np.any(color_arr == 'RED'):
    #         tracked_cells_df.COLOR[np.where(tracked_cells_df.SERIES == cell_num)[0]] = 'RED'
    #         num_RED += 1
            
    #     elif np.any(color_arr == 'YELLOW'):
    #         tracked_cells_df.COLOR[np.where(tracked_cells_df.SERIES == cell_num)[0]] = 'YELLOW'
    #         num_YELLOW += 1
        
    
    
    """ Pre-save everything """
    # tracked_cells_df = tracked_cells_df.sort_values(by=['SERIES', 'FRAME'])
    # tracked_cells_df.to_csv(sav_dir + 'tracked_cells_df_RAW.csv', index=False)
    
    
    ### (2) remove everything only on a single frame, except for very first frame
    # singles = []
    # for cell_num in np.unique(tracked_cells_df.SERIES):
          
    #            track_length_SEG = len(np.unique(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME))         


    #            """ remove anything that's only tracked for length of 1 timeframe """
    #            """ excluding if that timeframe is the very first one OR the very last one"""
               
    #            #print(truth_output_df.FRAME[truth_output_df.SERIES == cell_num] )
    #            if len(np.where(np.asarray(track_length_SEG) == 1)[0]) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == 0) and not np.any(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME == np.max(tracked_cells_df.FRAME)):
    #                singles.append(cell_num)
    #                tracked_cells_df = tracked_cells_df.drop(tracked_cells_df.index[np.where(tracked_cells_df.SERIES == cell_num)])
    #                continue;

    """  Save images in output """
    # input_name = examples[0]['input']
    # filename = input_name.split('/')[-1]
    # filename = filename.split('.')[0:-1]
    # filename = '.'.join(filename)
    
    # for frame_num, im_dict in enumerate(examples):
         
    #      output_frame = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im)
    #      im = convert_matrix_to_multipage_tiff(output_frame)
    #      imsave(sav_dir + filename + '_' + str(frame_num) + '_output.tif', im)
         
         
    #      output_frame = gen_im_frame_from_array(tracked_cells_df, frame_num=frame_num, input_im=input_im, color=1)
    #      im = convert_matrix_to_multipage_tiff(output_frame)
    #      imsave(sav_dir + filename + '_' + str(frame_num) + '_output_COLOR.tif', im)


    #      """ Also save image with different colors for RED/YELLOW and GREEN"""    
    #  ### (3) drop other columns
    # tracked_cells_df = tracked_cells_df.drop(columns=['visited', 'coords'])
    
    
    # ### and reorder columns
    # cols =  ['SERIES', 'COLOR', 'FRAME', 'X', 'Y', 'Z']
    # tracked_cells_df = tracked_cells_df[cols]

    # ### (4) save cleaned
    # tracked_cells_df.to_csv(sav_dir + 'tracked_cells_df_clean.csv', index=False)               
              
    list_exclude = []
            
    """ If want to reload quickly without running full analysis """            
    tracked_cells_df = pd.read_csv(sav_dir + 'tracked_cells_df_clean.csv', sep=',')           
    truth_output_df = pd.read_csv(sav_dir + 'truth_output_df.csv', sep=',')            
    truth_array = pd.read_csv(sav_dir + 'truth_array.csv', sep=',')               
    
    ### ALSO NEED LIST_EXCLUDE???      
            
    print('duplicates: ' + str(np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME']))))    ### *** REAL DUPLICATES
    tracked_cells_df.iloc[np.where(tracked_cells_df.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]]
       
            
    """ Plot compare to truth """
    if truth:
        truth_array.to_csv(sav_dir + 'truth_array.csv', index=False)
        truth_output_df = truth_output_df.sort_values(by=['SERIES'])
        truth_output_df.to_csv(sav_dir + 'truth_output_df.csv', index=False)
     
            
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
                        
                        
    
                        
        #plt.figure(); plt.plot(all_lengths)
        print(len(all_lengths))
        print(len(np.where(np.asarray(all_lengths) > 0)[0]))
        print(len(np.where(np.asarray(all_lengths) < 0)[0]))
        #truth_output_df = truth_output_df.sort_values(by=['SERIES'])
    
        """ Sort by errors """
        # sorted_lens = np.sort(all_lengths)
        # plt.figure(); 
        # plt.plot(sorted_lens)
        
        
        ### Figure out proportions
        total = len(all_lengths)
        prop = []; track_diff = [];
        uniques = np.unique(all_lengths)
        for num in uniques:
            len_num = len(np.where(all_lengths == num)[0])
            
            if len(prop) == 0:
                prop.append(0)
                prop.append(len_num/total)
                
                
            else:
                prop.append(prop[-1])
                prop.append(len_num/total + prop[-2])
            
            track_diff.append(num)
            track_diff.append(num)
            
        plt.figure(); plt.plot(prop, track_diff)
        ax = plt.gca()
        rs = ax.spines["right"]; rs.set_visible(False)
        ts = ax.spines["top"]; ts.set_visible(False)
        ax.margins(x=0)
        ax.margins(y=0.02)
        
        plt.xlabel("proportion of tracks", fontsize=14)
        plt.ylabel("track difference (# frames)", fontsize=14)
        













        """ Load old .csv FROM MATLAB OUTPUT and plot it??? 
        """
        MATLAB_name = 'MOBPF_190627w_5_output_FULL_AUTO_MATLAB.csv'

        #MATLAB_name = 'MOBPF_190626w_4_10x_output_PYTORCH_output_FULLY_AUTO.csv'
        #MATLAB_name = 'output.csv'
        
        
        MATLAB_auto_array = pd.read_csv(input_path + MATLAB_name, sep=',')
        
        
        all_cells_MATLAB = np.unique(MATLAB_auto_array.SERIES)
        all_cells_TRUTH = np.unique(truth_array.SERIES)
    
    
        all_lengths_MATLAB = []
        truth_lengths = []    
        MATLAB_lengths = []
        
        
        all_cell_nums = []
        for frame_num in range(len(examples)):
             print('Starting inference on volume: ' + str(frame_num) + ' of total: ' + str(len(examples)))
           
             seg_name = examples[frame_num]['seg']  
             seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
             seg = seg[0:lowest_z_depth, ...]
             seg = np.moveaxis(seg, 0, -1)       
             
             
        
             truth_next_im, truth_array  = gen_truth_from_csv(frame_num=frame_num, input_path=input_path, filename=truth_name, 
                                       input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
                              
             
             MATLAB_next_im, MATLAB_auto_array  = gen_truth_from_csv(frame_num=frame_num, input_path=input_path, filename=MATLAB_name, 
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
                       
                            all_lengths_MATLAB.append(track_length_TRUTH - track_length_MATLAB)
                            truth_lengths.append(track_length_TRUTH)
                            MATLAB_lengths.append(track_length_MATLAB)   
                            
                            all_cell_nums.append(num_new_truth)
                       
                       
        #plt.figure(); plt.plot(all_lengths)
        print(len(np.where(np.asarray(all_lengths_MATLAB) > 0)[0]))
        print(len(np.where(np.asarray(all_lengths_MATLAB) < 0)[0]))         
                                       
        """ Sort by errors """
        #sorted_lens_CNN = np.sort(all_lengths)
        #plt.figure(); 
        #plt.plot(sorted_lens_CNN)
        

        ### Figure out proportions
        total = len(all_lengths_MATLAB)
        prop = []; track_diff = [];
        uniques = np.unique(all_lengths_MATLAB)
        for num in uniques:
            len_num = len(np.where(all_lengths_MATLAB == num)[0])
            
            if len(prop) == 0:
                prop.append(0)
                prop.append(len_num/total)
                
                
            else:
                prop.append(prop[-1])
                prop.append(len_num/total + prop[-2])
            
            track_diff.append(num)
            track_diff.append(num)
            
        plt.plot(prop, track_diff)
        plt.savefig(sav_dir + 'plot.png')

        
        """ Parse the old array: """
        print('duplicates: ' + str(np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'FRAME']))))    ### *** REAL DUPLICATES
        MATLAB_auto_array.iloc[np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'FRAME'], keep=False))[0]]
        
        #print('duplicates: ' + str(np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'SERIES']))))   ### cell in same location across frames
        #MATLAB_auto_array.iloc[np.where(MATLAB_auto_array.duplicated(subset=['X', 'Y', 'Z',  'SERIES'], keep=False))[0]]
        
                   #duplicates: 378  
        
    
         
        """ **** SEE ONLY THE CELLS THAT WERE TRACKED IN MATLAB, TRUTH, and CNN!!! """
        """ Things to fix still:
    
             (2) ***blobs
             
             
             does using del [] on double tracked cells do anything bad???
        
            ***FINAL OUTPUT:
                    - want to show on tracked graph:
                            (a) cells tracked over time, organize by mistakes highest at top horizontal bar graph ==> also only used cells matched across all 3 matrices
                            (b) show number of cells tracked by each method
                            (c) show # of double_linked to be resolved
                            
        
        
        """
        
        
        
        """ Scale x-axis so is % of tracked successfully """
        
    


        """ Plot """
        
        
        def plot_timeframes(tracked_cells_df, add_name='OUTPUT_'):
            zeros = np.zeros(np.shape(input_im))
            
            
            new_cells_per_frame =  np.zeros(len(np.unique(tracked_cells_df.FRAME)))
            terminated_cells_per_frame =  np.zeros(len(np.unique(tracked_cells_df.FRAME)))
            num_total_cells_per_frame = np.zeros(len(np.unique(tracked_cells_df.FRAME)))
            
            
            num_large = 0
            cell_nums = []
            for cell_num in np.unique(tracked_cells_df.SERIES):
                
                frames_cur_cell = tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].FRAME
                
                beginning_frame = np.min(frames_cur_cell)
                if beginning_frame > 0:   # skip the first frame
                    new_cells_per_frame[beginning_frame] += 1
                    

                
                
                #if beginning_frame == 1:
                    
                    #if tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].Z.iloc[0] >= 125:
                        #print('bottom to eliminate')
                    #else:
                            
                        #print(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].Z)
                        #num_large += 1
                    #    num = 1
                    
                            
                term_frame = np.max(frames_cur_cell)
                if term_frame < len(terminated_cells_per_frame) - 1:   # skip the last frame
                    terminated_cells_per_frame[term_frame + 1] += 1
                    
                    if term_frame == 2:
                        #and tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].Z.iloc[-1]
                        print(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].Z)
                        print(len(np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].coords)[0]))
                        
                        
                        coords = np.asarray(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].coords)[0]
                        color = tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].COLOR
                        print(color)
                             
                        
                        
                        
                        if len(coords) > 300:
                        
                            num_large += 1 
                            cell_nums.append(tracked_cells_df.iloc[np.where(tracked_cells_df.SERIES == cell_num)].index)
                            zeros[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
                    
                
                for num in frames_cur_cell:
                    num_total_cells_per_frame[num] += 1    
                
                
                
    
    
            y_pos = np.unique(tracked_cells_df.FRAME)
            plt.figure(); plt.bar(y_pos, new_cells_per_frame, color='k')
            ax = plt.gca()
            rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
            name = 'new cells per frame'
            #plt.title(name);
            plt.xlabel('time frame', fontsize=16); plt.ylabel('# new cells', fontsize=16)
            # ax.set_xticklabels(x_ticks, rotation=0, fontsize=12)
            # ax.set_yticklabels(y_ticks, rotation=0, fontsize=12)
            plt.savefig(sav_dir + add_name + name + '.png')
    
            plt.figure(); plt.bar(y_pos, terminated_cells_per_frame, color='k')
            ax = plt.gca()
            rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
            name = 'terminated cells per frame'
            #plt.title(name)
            plt.xlabel('time frame', fontsize=16); plt.ylabel('# terminated cells', fontsize=16)
            plt.savefig(sav_dir + add_name + name + '.png')
    
            
            plt.figure(); plt.bar(y_pos, num_total_cells_per_frame, color='k')
            ax = plt.gca()
            rs = ax.spines["right"]; rs.set_visible(False); ts = ax.spines["top"]; ts.set_visible(False)
            name = 'number cells per frame'
            #plt.title(name)
            plt.xlabel('time frame', fontsize=16); plt.ylabel('# cells', fontsize=16)
            plt.savefig(sav_dir + add_name + name + '.png')



    """ plot timeframes """
    plot_timeframes(tracked_cells_df, add_name='OUTPUT_')
    plot_timeframes(MATLAB_auto_array, add_name='MATLAB_')
    plot_timeframes(truth_array, add_name='TRUTH_')
    
    
    
    
    
    
    
    
    
    
    