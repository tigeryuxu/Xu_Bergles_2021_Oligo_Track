%% Need to add for 3D:
% 1) ridges2lines ==> needs to separate into 3 types of angles (x,y,z) ==>
% OR just take it out completely???
% 2) must fix rest of code to adapt to nanofiber cultures
% 3) must fix all "disk" dilations to "spheres"



% IF WANT TO ADJUST/Elim background, use ImageJ
% ==> 1) Split channels, 2) Select ROI, 3) go to "Edit/Clear outside"
% 4) Merge Channels, 5) Convert "Stack to RGB", 6) Save image

%***Note for Annick: ==> could also use ADAPTHISTEQ MBP for area at end...
%but too much???

% ***ADDED AN adapthisteq to imageAdjust.mat... 2019-01-24

%% Main function to run heuristic algorithm
opengl hardware;
close all;

cur_dir = pwd;
addpath(strcat(cur_dir))  % adds path to functions
addpath('../IO_func/')
addpath('../man_corr_func/')
addpath('../watershed_func/')
addpath('../cell_crop_func/')
cd(cur_dir);

%% Initialize
foldername = uigetdir();   % get directory

%% Run Analysis
cd(foldername);   % switch directories
nameCat = '*tif*';
fnames = dir(nameCat);

trialNames = {fnames.name};
numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently
natfnames=natsort(trialNames);
%% Read in images
empty_file_idx_sub = 0;
for fileNum = 1 : 3: numfids
    
    cd(cur_dir);
    
    filename_raw_gray = natfnames{fileNum};
    cd(foldername);
    [gray] = load_3D_gray(filename_raw_gray);
    plot_max(gray);
    
    
    filename_raw_ilastik = natfnames{fileNum + 1};
    cd(foldername);
    [ilastik] = load_3D_gray(filename_raw_ilastik);
 
    ilastik(ilastik > 0.007) = 0;
       ilastik(ilastik > 0) = 1;
       
    %gray = gray(:, :, 100:125);
    plot_max(ilastik);
    
    cd(cur_dir);
    filename_raw_truth = natfnames{fileNum + 2};
    cd(foldername);
    [truth] = load_3D_gray(filename_raw_truth);
    truth(truth > 0) = 255;
    plot_max(truth);
    

    cc = bwconncomp(ilastik);
    ilastik_regions_table = regionprops3(cc, gray, 'Volume','Centroid', 'VoxelIdxList', 'VoxelValues', 'MeanIntensity'); %%%***good way to get info about region!!!

    cc = bwconncomp(truth);
    truth_table = regionprops3(cc, gray, 'Volume','Centroid', 'VoxelIdxList', 'VoxelValues', 'MeanIntensity'); %%%***good way to get info about region!!!

    
    new_truth_im = zeros(size(truth));
    for cell_idx = 1:length(truth_table.VoxelIdxList)
        truth_cell = truth_table.VoxelIdxList{cell_idx};
        
        matched = 0;
        for ilastik_idx = 1:length(ilastik_regions_table.VoxelIdxList)
            ilastik_cell = ilastik_regions_table.VoxelIdxList{ilastik_idx};
            same = ismember(truth_cell, ilastik_cell);
            
            %% if matched
            if ~isempty(find(same, 1))
                
                matched = 1;
                new_truth_im(ilastik_cell) = 1;
                break;
            end
        end
        
        %% otherwise, add the original cell back in
        if matched == 0
            new_truth_im(truth_cell) = 1;
        end
    end
    plot_max(new_truth_im);
    
      %% TIGER - REMOVED - May 22nd 2020
    %% Also keep all ILASTIK things near bottom of volume
%     last_keep_slice = 100;
%     for bottom_idx = 1:length(ilastik_regions_table.VoxelIdxList)
%        if ilastik_regions_table.Centroid(bottom_idx, 3) > last_keep_slice
%            
%            ilastik_cell = ilastik_regions_table.VoxelIdxList{bottom_idx};
%            new_truth_im(ilastik_cell) = 1;
%    
%        end
%     end
%     plot_max(new_truth_im);
% 
%     new_truth_im(new_truth_im > 0) = 255;
%     new_truth_im = uint8(new_truth_im);
    
    
    
    
    %% normalize labels to be between 1 - 3
    %labels_norm = mod(labels, 15) 1;
    %labels_norm(labels == 0) = 0;
    %labels = labels;
    
    mkdir('output');
    cd('./output')
    
    z_size = length(new_truth_im(1, 1, :));
    filename_raw_truth_ilastik = filename_raw_truth(1:end-5);
    for k = 1:z_size
        input = new_truth_im(:, :, k);
        imwrite(input, strcat(filename_raw_truth_ilastik,'_m_ilastik.tif') , 'writemode', 'append', 'Compression','none')
    end
    
    
    
    %% Save max projects as well
    axis = 3;
    gray_max = max(gray, [], axis);
    filename_raw_gray = filename_raw_gray(1:end-4);
    imwrite(gray_max, strcat(filename_raw_gray,'_max.tif') , 'writemode', 'append', 'Compression','none')
    
    
    truth_max = max(truth, [], axis);
    filename_raw_truth = filename_raw_truth(1:end-4);
    imwrite(truth_max, strcat(filename_raw_truth,'_max.tif') , 'writemode', 'append', 'Compression','none')
    
    
    ilastik_max = max(ilastik, [], axis);
    filename_raw_ilastik = filename_raw_ilastik(1:end-5);
    imwrite(ilastik_max, strcat(filename_raw_ilastik,'_max.tif') , 'writemode', 'append', 'Compression','none')
    
    new_truth_im_max = max(new_truth_im, [], axis);
    imwrite(new_truth_im_max, strcat(filename_raw_truth_ilastik,'_m_ilastik_max.tif') , 'writemode', 'append', 'Compression','none')
        
    
    
    
    close all;
    
    
    
    
end



