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
addpath('./IO_func/')
addpath('./man_corr_func/')
addpath('./watershed_func/')
addpath('./cell_crop_func/')
cd(cur_dir);

%% Initialize
foldername = uigetdir();   % get directory

%% Run Analysis
cd(foldername);   % switch directories
nameCat = '*tif*';
fnames = dir(nameCat);

trialNames = {fnames.name};
numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently

%% Read in images
empty_file_idx_sub = 0;
for fileNum = 1 : numfids
    
        cd(cur_dir);
        natfnames=natsort(trialNames);
        filename_raw = natfnames{fileNum};
        %% Decide if want to load individual channels or single image
        cd(foldername);
        [gray] = load_3D_gray(filename_raw);
      
        %gray = gray(:, :, 100:125);
        figure(1); volshow(gray);
        
        %% TIGER - CHANGED "enhance", "Human_OL", and "cropping size" ==> all for Daryan's stuff
        DAPIsize = 0;
        DAPImetric = 0;
        enhance_DAPI = 'N'
        DAPI_bb_size = 50
        binary = 'Y'
        [mat, objDAPI, DAPI_bw, labels] = DAPIcount_3D(gray, DAPIsize, DAPImetric, enhance_DAPI, DAPI_bb_size, binary);  % function
       
        %% Initializes struct to store everything
        c= cell(length(objDAPI), 1); % initializes Bool_W with all zeros
        [c{:}] = deal(0);
        strucMat = num2cell(mat, 2);
        s = struct('objDAPI', objDAPI', 'centerDAPI', strucMat, 'Core', cell(length(objDAPI), 1)...
            ,'CB', cell(length(objDAPI), 1), 'Fibers', cell(length(objDAPI), 1), 'Mean_Fiber_L_per_C', cell(length(objDAPI), 1), 'Bool_W', c...
            , 'im_num', c, 'O4_bool', c, 'AreaOverall', c, 'numO4', c, 'im_size', c, 'OtherStats', c);
            
        %% save input images
        
        
        labels = uint8(labels);
        %% normalize labels to be between 1 - 3
        %labels_norm = mod(labels, 15) + 1;
        %labels_norm(labels == 0) = 0;
        %labels = labels;
        
        z_size = length(labels(1, 1, :));
        for k = 1:z_size
            input = labels(:, :, k);
            imwrite(input, strcat(filename_raw,'_watershed_seg.tif') , 'writemode', 'append', 'Compression','none')
        end
end



