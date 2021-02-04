#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:09:12 2020

@author: user
"""
from functions_cell_track_auto import *


tracked_cells_df, recheck_series, next_seg = clean_with_predictions(tracked_cells_df, new_candidates, next_seg, crop_size, z_size, frame_num, seg_train, height_tmp, width_tmp, depth_tmp, min_dist=10)
