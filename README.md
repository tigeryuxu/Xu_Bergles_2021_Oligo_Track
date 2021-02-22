# Oligo-Track installation and demo

### README file under construction ==> email: yxu130@jh.edu for detailed inquiries

### Installation:
* 1. clone this git repo
```
git clone https://github.com/yxu233/Xu_Bergles_2021_Oligo_Track.git
```

* 2. install dependencies in virtual environment:
```
cd Xu_Bergles_2021_Oligo_Track
virtualenv -p python3.7 venv
source venv/bin/activate
pip install natsort skimage tifffile tkinter pytorch
```

* 3. install Pytorch from https://pytorch.org/. Choose correct:
	- version 1.71
	- CUDA version that is compatible with GPU
	- Proper operating system
* 3. download checkpoint files from:  https://www.dropbox.com/sh/lzokf3sd0diuyn3/AABYV_oQxo0a-vAy_e_xIYEea?dl=0
* 4. place checkpoint files into the 'Checkpoints' folder


### Other requirements:
* 1. GPU with at least 4 GB RAM
* 2. CPU with at least 16 GB RAM
   
### Setup data for analysis:
* overall, images should resemble input in the "demo data" folder
* individual multipage tiff stacks for each timepoint
* each stack must be single channel, grayscale, uint8 and named in order of timeseries
   
### Analysis pipeline overview:
* 1. Run Seg_CNN_inference.py. Select input folders as indicated by GUI.
* 2. Run Track_CNN_inference.py. The input folder for Track-CNN is the output folder of Seg-CNN.

Note: Seg and Track CNN are run separately at this time to enable parallelization on separate computers as needed


### Demo run:
* 1. Run Seg_CNN_inference.py and select the "demo data" folder as the input path
* 2. After step 1 is done, run Track_CNN_inference.py and select the output folder from step 1 as the new input folder. 
 
