# Bergles_cell_tracking_semi_auto

### Installation: (ensure > MATLAB 2019a, or else sliceViewer will not work)
* 1. Install github desktop: https://desktop.github.com/
* 2. Clone the git repo so it shows up in github desktop
   
   
### Analysis pipeline:
* 1. Run Ilastik on raw data to get cell body segmentations
* 2. Run Bergles_watershed_and_save.m on the Ilastik data to separate adjacent cell bodies
* 3. Place both the RAW data (fluorescence) and watershed identified cell bodies into a folder so the files are interleaved (time_1_raw, time_1_cell_bodies, time_2_raw, time_2_cell_bodies, ect...)
* 4. Run Bergles_cell_tracking_semi_auto.m to start analysis


### Manual analysis hot-keys: - if want new hotkeys, modify "Bergles_manual_count.m"
* "1" - classify as same tracked cell
* "2" - classify as no longer tracked
* "a" - add
* "s" - scale
* "d" - delete cell (from current frame) permanently - i.e. if cell body is junk
* "c" - CLAHE
* "h" - hide cell body overlay

### GUI inputs (and their defaults):
* Crop_size (XY pixels): 200 - defines X and Y lengths for cropping
* z_size (Z pixels): 25 - defines axial depth in crops
* ssim_thresh (0 - 1): 0.30 - defines threshold for SSIM used to identify highly-confident pairs
* low_dist_thresh (0 - 20): 15 - defines euclidean distance threshold for highly-confident pairs (must be super close)
* upper_dist_thresh (30 - 100): 25 - defines upper threshold for confidently NOT paired (super far)
* min_size (0 - 500 pixels): 200 - defines minimum size of cell bodies to be tracked
* first_slice: 10 - first slice of image to analyze
* last_slice: 110 - last slice of image to analyze (overall ~100 slices == 300 um)
* manual_correct? (Y/N): Y - defines if want to do full auto or semi-auto analysis


### Demo run:
* 1. Download data here (5 GB), includes output data from my corrections: https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/yxu130_jh_edu/EpeTEaEYmB5FvK4ESh-uv7oB3cjEyifYDWRSDdLitczvow?e=VHYxb2 
 
* Specifically, take a look at the file "Merged.tif" in folder "Outputs attempt 2", which has 2 channels (raw data + tracked outputs) along all timeframes for 300 um (100 slices). NOTE: the colors in the 2nd tracked outputs corresponds to the SAME color for each tracked cell across timeframes, so if you scroll across timeframes and see the same color, that color corresponds to the same cell being tracked.
   
   
### Usage notes:
* For manual correction, ensure that you press the button twice (once to remove "pause" in MATLAB, and once to actually select option)
* don't press cancel on anything... might crash

   
