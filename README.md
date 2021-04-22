# Oligo-Track installation and demo

### Installation:
1. clone this git repo
```
git clone https://github.com/yxu233/Xu_Bergles_2021_Oligo_Track.git
cd Xu_Bergles_2021_Oligo_Track
```


2. Create virtual environment:

- Linux:
```
virtualenv -p python3.7 venv
source venv/bin/activate
```

- Windows: (recommended download anaconda https://www.anaconda.com/products/individual#windows, and then use anaconda command prompt to make virtualenv)
```
conda update conda
conda create -n venv python=3.7 anaconda
conda activate venv
```

3. Install dependencies:
```
pip install numpy matplotlib natsort scikit-image tifffile pandas PySimpleGUI sklearn progressbar2 seaborn
```

4. install Pytorch from https://pytorch.org/. Choose correct:
	- version 1.71
	- CUDA version that is compatible with GPU
	- Proper operating system

5. download checkpoint files from:  https://www.dropbox.com/sh/lzokf3sd0diuyn3/AABYV_oQxo0a-vAy_e_xIYEea?dl=0 and place the checkpoint files into the 'Checkpoints' folder


### Other requirements:
1. GPU with at least 6 GB RAM
2. CPU with at least 16 GB RAM
   
### Setup data for analysis:
* overall, images should resemble input in the "demo data" folder.
* Each timepoint should have corresponding individual multipage tiff stack
* Each tiff stack must be single channel, grayscale, uint8 and named in ascending order of timeseries
* Image pixel scale is VERY important! The expected scaling is: 0.83 um/px in XY and 3 um/px in Z. To avoid segmentation artifacts, please down- or up-sample your data appropriately to match this scaling.
   
### Analysis pipeline overview:
1. Run Seg_CNN_inference.py. Select input folders as indicated by GUI. (*remember to activate virtual environment as above)
```
cd Oligo_Track
python Seg_CNN_inference.py
```

2. Run Track_CNN_inference.py. The input folder for Track-CNN is the output folder of Seg-CNN.
```
python Track_CNN_inference.py
```

Note: Seg and Track CNN are run separately at this time to enable parallelization on separate computers as needed


### Demo run:
1. Download demo data from https://www.dropbox.com/sh/kebq4sypu6zkdrb/AAALicP6Ia4p8hLJ_3-1AbDaa?dl=0
2. Run Seg_CNN_inference.py and select the "demo data" folder as the input path
3. After step 1 is done, run Track_CNN_inference.py and select the output folder from step 1 as the new input folder.


### Output files:
1. For Seg-CNN: output files end in "_segmentation.tif" and contain the volumetric binary segmentation of cells in the corresponding raw data
2. For Track-CNN:
	- _POST_PROCESSED.csv: each row corresponds to 1 cell on 1 timeseries, indicating XYZt coordinates, volume in um^3
	- _POST_PROCESSED_for_SYGLASS.csv: file can be loaded directly into Syglass VR software
	- _POST_PROCESSED_pickle.pkl: same data but in pickle format for python. Additionally, each cell row also has a list of coordinates for the entire cell soma.
	- CLEANED.tif: contains tracked cells, where each cell soma is labeled with a different value that corresponds to the same cell across multiple timepoints.
 	- .png files: simple plots of tracked dynamics over timeseries. Each plot is normalized to baseline (1st timeframe). Plots from across entire volumes, or slices 0 - 32, 33 - 65, 66 - 99, 100 - 132, 133 - 165
	- additional statistics/analysis will require additional custom written code. Please reach out to yxu130@jhmi.edu if you require help with setting up additional analysis
	



 
