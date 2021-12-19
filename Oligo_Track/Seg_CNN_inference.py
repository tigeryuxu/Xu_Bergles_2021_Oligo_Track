# -*- coding: utf-8 -*-
"""
Seg-CNN:
    Expected input:
        - series of Tiffs
        
        
        - need GPU with at least 6 GB RAM
        
        

"""

import sys
sys.path.insert(0, './layers')


import glob, os

os_windows = 0
if os.name == 'nt':  ## in Windows
     os_windows = 1;
     print('Detected Microsoft Windows OS')
else: print('Detected non-Windows OS')


import numpy as np
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import torch
#from UNet_pytorch_online import *
from layers.tracker import *

from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.UNet_functions_PYTORCH import *
from functional.GUI import *
import tifffile as tiff

from layers.UNet_pytorch_online import *

from skimage.filters import threshold_otsu
from skimage.filters import threshold_triangle
from skimage.transform import rescale, resize, downscale_local_mean

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True



""" Define GPU to use """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""  Network Begins: """
#s_path = './(21) Checkpoints_PYTORCH_NO_transforms_AdamW_batch_norm_CLEAN_DATA_LARGE_NETWORK/'
s_path = './Checkpoints/'

overlap_percent = 0.5
input_size = 256
depth = 64
num_truth_class = 2

XY_expected = 0.83; Z_expected = 3;

""" TO LOAD OLD CHECKPOINT """
# Read in file names
onlyfiles_check = glob.glob(os.path.join(s_path,'Seg_CNN_check_*'))
onlyfiles_check.sort(key = natsort_key1)
      
last_file = onlyfiles_check[-1]
split = last_file.split('check_')[-1]
num_check = split.split('.')
checkpoint = num_check[0]
checkpoint = 'Seg_CNN_check_' + checkpoint
check = torch.load(s_path + checkpoint, map_location=device)
tracker = check['tracker']

#unet = check['model_type']; 
""" Initialize network """  
kernel_size = 5
pad = int((kernel_size - 1)/2)
unet = UNet_online(in_channels=1, n_classes=2, depth=5, wf=4, kernel_size = kernel_size, padding= int((kernel_size - 1)/2), 
                    batch_norm=True, batch_norm_switchable=False, up_mode='upsample')

unet.load_state_dict(check['model_state_dict'])
unet.to(device); unet.eval()
print('parameters:', sum(param.numel() for param in unet.parameters()))


""" Select multiple folders for analysis AND creates new subfolder for results output """
list_folder, XY_res, Z_res = seg_CNN_GUI()

""" Loop through all the folders and do the analysis!!!"""
for input_path in list_folder:
    foldername = input_path.split('/')[-2]
    sav_dir = input_path + '/' + foldername + '_output_seg_CNN'

    """ For testing ILASTIK images """
    images = glob.glob(os.path.join(input_path,'*.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('.tif','_truth.tif'), ilastik=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]
     
    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("\nSave directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("\nSave directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'
    
    # Required to initialize all
    for i in range(len(examples)):
         
        
         """ TRY INFERENCE WITH PATCH-BASED analysis from TORCHIO """
         with torch.set_grad_enabled(False):  # saves GPU RAM            
            input_name = examples[i]['input']  
            input_im = tiff.imread(input_name)
            
            
            
            
            # import SimpleITK as sitk
            # elastixImFilter = sitk.ElastixImageFilter()
            # elastixImFilter.SetFixedImage(sitk.ReadImage(examples[i]['input']))
            # elastixImFilter.SetMovingImage(sitk.ReadImage(examples[i + 1]['input']))
            
            # parameterMapVector = sitk.VectorOfParameterMap()
            # parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
            # parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
            
            # elastixImageFilter.SetParameterMap(parameterMapVector)
            
            # elastixImageFilter.Execute()
            # sitk.WriteImage(elastixImageFilter.GetResultImage())
            
            
            
            # zzz
            
            
            
            
#             import SimpleITK as sitk
            
#             def command_iteration(filter):
#                 print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
            
#             fixed = 

#             moving = sitk.ReadImage(examples[i + 1]['input'], sitk.sitkFloat32)
            
#             matcher = sitk.HistogramMatchingImageFilter()
#             matcher.SetNumberOfHistogramLevels(1024)
#             matcher.SetNumberOfMatchPoints(7)
#             matcher.ThresholdAtMeanIntensityOn()
#             moving = matcher.Execute(moving, fixed)
            
#             #zzz
#             # The basic Demons Registration Filter
#             # Note there is a whole family of Demons Registration algorithms included in
#             # SimpleITK
#             demons = sitk.DemonsRegistrationFilter()
#             demons.SetNumberOfIterations(500)
#             # Standard deviation for Gaussian smoothing of displacement field
#             demons.SetStandardDeviations(1.0)
            
#             demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))
            
#             displacementField = demons.Execute(fixed, moving)
            
#             print("-------")
#             print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
#             print(f" RMS: {demons.GetRMSChange()}")
            
#             outTx = sitk.DisplacementFieldTransform(displacementField)
            
#             sitk.WriteTransform(outTx, input_name + '_output.hdf5')
            
#             if ("SITK_NOSHOW" not in os.environ):
#                 resampler = sitk.ResampleImageFilter()
#                 resampler.SetReferenceImage(fixed)
#                 resampler.SetInterpolator(sitk.sitkLinear)
#                 resampler.SetDefaultPixelValue(100)
#                 resampler.SetTransform(outTx)
            
#                 out = resampler.Execute(moving)
#                 simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
#                 simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
#                 # Use the // floor division operator so that the pixel type is
#                 # the same for all three images which is the expectation for
#                 # the compose filter.
#                 cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
#                 sitk.Show(cimg, "DeformableRegistration1 Composition")
                        
                
#             arr1 = sitk.GetArrayFromImage(simg1)
#             arr2 = sitk.GetArrayFromImage(simg2)
#             out_arr = sitk.GetArrayFromImage(cimg)
                





# def demons_registration(fixed_image, moving_image, fixed_points = None, moving_points = None):
    
#     registration_method = sitk.ImageRegistrationMethod()

#     # Create initial identity transformation.
#     transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
#     transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
#     # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
#     initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sitk.Transform()))
    
#     # Regularization (update field - viscous, total field - elastic).
#     initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0) 
    
#     registration_method.SetInitialTransform(initial_transform)

#     registration_method.SetMetricAsDemons(10) #intensities are equal if the difference is less than 10HU
        
#     # Multi-resolution framework.            
#     registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
#     registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8,4,0])    

#     registration_method.SetInterpolator(sitk.sitkLinear)
#     # If you have time, run this code as is, otherwise switch to the gradient descent optimizer    
#     #registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
#     registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
#     registration_method.SetOptimizerScalesFromPhysicalShift()

#     # If corresponding points in the fixed and moving image are given then we display the similarity metric
#     # and the TRE during the registration.
#     if fixed_points and moving_points:
#         registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
#         registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)        
#         registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points, moving_points))
        
#     return registration_method.Execute(fixed_image, moving_image)    

# #%%timeit -r1 -n1
# # Uncomment the line above if you want to time the running of this cell.

# # Select the fixed and moving images, valid entries are in [0,9]
# fixed_image_index = 0
# moving_image_index = 7


# tx = demons_registration(fixed, moving, fixed_points = None, moving_points = None)

# initial_errors_mean, initial_errors_std, _, initial_errors_max, initial_errors = ru.registration_errors(sitk.Euler3DTransform(), points[fixed_image_index], points[moving_image_index])
# final_errors_mean, final_errors_std, _, final_errors_max, final_errors = ru.registration_errors(tx, points[fixed_image_index], points[moving_image_index])

# plt.hist(initial_errors, bins=20, alpha=0.5, label='before registration', color='blue')
# plt.hist(final_errors, bins=20, alpha=0.5, label='after registration', color='green')
# plt.legend()
# plt.title('TRE histogram');
# print('Initial alignment errors in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(initial_errors_mean, initial_errors_std, initial_errors_max))
# print('Final alignment errors in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(final_errors_mean, final_errors_std, final_errors_max))






            """ Scale images to default resolution if user resolution does not matching training """
            XY_scale = float(XY_res)/XY_expected
            if XY_scale < 1: print('Image XY resolution does not match training resolution, and will be downsampled by: ' + str(round(1/XY_scale, 2)))
            elif  XY_scale > 1: print('Image XY resolution does not match training resolution, and will be upsampled by: ' + str(round(XY_scale, 2)))


            Z_scale = float(Z_res)/Z_expected
            if Z_scale < 1: print('Image Z resolution does not match training resolution, and will be downsampled by: ' + str(round(1/Z_scale, 2)))
            elif  Z_scale > 1: print('Image Z resolution does not match training resolution, and will be upsampled by: ' + str(round(Z_scale, 2)))
            
            
            if XY_scale != 1 or Z_scale != 1:
                input_im = rescale(input_im, [Z_scale, XY_scale, XY_scale], anti_aliasing=True, order=2, preserve_range=True)   ### rescale the images
                    
                input_im = ((input_im - input_im.min()) * (1/(input_im.max() - input_im.min()) * 255)).astype('uint8')   ### rescale to 255


            """ Analyze each block with offset in all directions """
            ### CATCH error if too small volume, need to pad with zeros!!!
            if input_im.shape[0] < depth: pad_z = depth 
            else: pad_z = input_im.shape[0] 

            if input_im.shape[1] < input_size: pad_x = input_size 
            else: pad_x = input_im.shape[1] 
            
            if input_im.shape[2] < input_size: pad_y = input_size 
            else: pad_y = input_im.shape[2]             
            
            pad_im = np.zeros([pad_z, pad_x, pad_y])
            pad_im[:input_im.shape[0], :input_im.shape[1], :input_im.shape[2]] = input_im
            input_im = pad_im
            
            
            """ Find reference free SNR """
            all_SNR = [];        
            thresh = threshold_otsu(input_im)
            for slice_depth in range(0, len(input_im) - 33, 33):
            
                first_slices= input_im[slice_depth:slice_depth + 33, ...]
                max_first = plot_max(first_slices, ax=0, plot=0)
                signal = np.mean(np.where(max_first > thresh))
                noise = np.std(np.where(max_first < thresh))
                SNR = 10 * math.log10(signal/noise)
                all_SNR.append(round(SNR, 3))
            all_SNR = np.asarray(all_SNR)
            below_thresh_SNR = np.where(all_SNR < 1.5)[0]
            if len(below_thresh_SNR) > 0:
                print('\nWARNING: SNR is low for image: ' + input_name + 
                      '\n starting at depth slice: ' + str(below_thresh_SNR * 33) + 
                      '\n with SNR values: ' + str(all_SNR[below_thresh_SNR]) )
            
            
        
            """ Start inference on volume """
            print('\nStarting inference on volume: ' + str(i + 1) + ' of total: ' + str(len(examples)))
            segmentation = UNet_inference_by_subparts_PYTORCH(unet, device, input_im, overlap_percent, quad_size=input_size, quad_depth=depth,
                                                      mean_arr=tracker.mean_arr, std_arr=tracker.std_arr, num_truth_class=num_truth_class,
                                                      skip_top=1)
           
            segmentation[segmentation > 0] = 255
            filename = input_name.split('/')[-1].split('.')[0:-1]
            filename = '.'.join(filename)
            
            ### if operating system is Windows, must also remove \\ slash
            if os_windows:
                 filename = filename.split('\\')[-1]
                 
                 
            segmentation = np.asarray(segmentation, np.uint8)
            tiff.imsave(sav_dir + filename + '_' + str(int(i)) +'_segmentation.tif', segmentation)
            segmentation[segmentation > 0] = 1
            
            input_im = np.asarray(input_im, np.uint8)
            tiff.imsave(sav_dir + filename + '_' + str(int(i)) +'_input_im.tif', input_im)
            
        
    print('\n\nSegmented outputs saved in folder: ' + sav_dir)