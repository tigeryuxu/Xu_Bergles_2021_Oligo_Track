# Oligo-Track installation and demo


### README file under construction ==> email: yxu130@jh.edu for inquiries

### Installation: (ensure > MATLAB 2019a, or else sliceViewer will not work)
* 1. clone this git repo
* 2. install dependencies:
			pip install 
			
* 3. download checkpoint files from: 
* 4. place checkpoint files into appropriate folders

   
### Analysis pipeline overview:
* 1. Run Seg_CNN_inference.py
* 2. Run Track_CNN_inference.py. The input folder for Track-CNN is the output folder of Seg-CNN.

Note: Seg and Track CNN are run separately at this time to enable parallelization on separate computers as needed


### Demo run:
* 1. Run Seg_CNN_inference.py and select the "demo data" folder as the input path
* 2. After step 1 is done, run Track_CNN_inference.py and select the output folder from step 1 as the new input folder. 
 
