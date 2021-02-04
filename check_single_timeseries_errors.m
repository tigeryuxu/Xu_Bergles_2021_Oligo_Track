%% Find cells that only exist on a single frame

opengl hardware;
close all;

addpath('./IO_func/')
addpath('./man_corr_func/')
addpath('./watershed_func/')
addpath('./cell_crop_func/')

cur_dir = pwd;
addpath(strcat(cur_dir))  % adds path to functions
cd(cur_dir);

%% Initialize
foldername = uigetdir();   % get directory

%% Run Analysis
cd(foldername);   % switch directories
nameCat = '*tif*';
fnames = dir(nameCat);

trialNames = {fnames.name};
numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently

cd(cur_dir);
natfnames=natsort(trialNames);


%% Input dialog values
prompt = {'crop size (XY px): ', 'z_size (Z px): ', 'ssim_thresh (0 - 1): ', 'low_dist_thresh (0 - 20): ', 'upper_dist_thresh (30 - 100): ', 'min_siz (0 - 500): ', 'first_slice: ', 'last_slice: ', 'manual_correct? (Y/N): '};
dlgtitle = 'Input';
definput = {'200', '25', '0.30', '15', '25', '200', '10', '110', 'Y'};
answer = inputdlg(prompt,dlgtitle, [1, 35], definput);

crop_size = str2num(answer{1})/2;
z_size = str2num(answer{2});
ssim_val_thresh = str2num(answer{3});
dist_thresh = str2num(answer{4});
upper_dist_thresh = str2num(answer{5});

thresh_size = str2num(answer{6}); % pixels at the moment
first_slice = str2num(answer{7});
last_slice = str2num(answer{8}); % 100 * 3 == 300 microns
manual_correct_bool = answer{9};

%% Load matrix timeseries
cd(foldername);
load matrix_timeseries


%% Find number of non-empty in this current row:
num_non_empty =sum(~cellfun(@isempty,matrix_timeseries),2);

%% Identify remaining unassociated cells and add them to the cell list with NEW numbering (at the end of the list)
matrix_timeseries_cleaned = matrix_timeseries;
disp('checking single cells')
all_s = cell(0);
timeframe_idx = 1;
count = 0
for sorted_idx = 1:2:length(natfnames) - 2
    
    fileNum = sorted_idx;
    [all_s, frame_1, truth_1] = load_data_into_struct(foldername, natfnames, fileNum, all_s, thresh_size, first_slice, last_slice);
    
    im_frame = zeros(size(frame_1));
    for cell_idx = 1:length(matrix_timeseries(:, 1))
        if isempty(matrix_timeseries{cell_idx, timeframe_idx})
            continue;
        end
        cur_cell = matrix_timeseries{cell_idx, timeframe_idx};
        next_cell = matrix_timeseries{cell_idx, timeframe_idx + 1};
        
        voxels = cur_cell.voxelIdxList;
        

        % If next timeseries is empty (AND timeframe_idx == 1 OR previous timeframe is also empty), means that cell was terminated early
        % so plot the output for the next one
        
        
        %if num_non_empty(cell_idx) == 1
            
        %if length(voxels) <= 300
        if timeframe_idx == 2 &&  isempty(matrix_timeseries{cell_idx, timeframe_idx - 1})
            count = count + 1;
            frame_1_centroid = cur_cell.centroid;
            %crop_size = 8;
            %z_size = 16;
            im_size = size(frame_1);
            height = im_size(1);  width = im_size(2); depth = im_size(3);
            y = round(frame_1_centroid(1)); x = round(frame_1_centroid(2)); z = round(frame_1_centroid(3));
            [crop_frame_1, box_x_min ,box_x_max, box_y_min, box_y_max, box_z_min, box_z_max] = crop_around_centroid(frame_1, y, x, z, crop_size, z_size, height, width, depth);
            
            blank_truth = zeros(size(truth_1));
            blank_truth(x, y, z) = 1;
            %blank_truth = imdilate(blank_truth, strel('sphere', 3));
            crop_blank_truth_1 = crop_around_centroid(blank_truth, y, x, z, crop_size, z_size, height, width, depth);
            crop_blank_truth_1 = imdilate(crop_blank_truth_1, strel('sphere', 3));
            
            mip_1 = max(crop_frame_1, [], 3);
            figure(1); imshow(mip_1);
            
            mip_center_1 = max(crop_blank_truth_1, [], 3);
            magenta = cat(3, ones(size(mip_1)), zeros(size(mip_1)), ones(size(mip_1)));
            hold on;
            h = imshow(magenta);
            hold off;
            set(h, 'AlphaData', mip_center_1)
            title(strcat('x: ', num2str(x), ' y: ', num2str(y), 'z: ', num2str(z)));
            %pause
            
            close all;
            
            im_frame(voxels) = 1;
            disp('yep')
            
            matrix_timeseries_cleaned{cell_idx, timeframe_idx} = [];
        end
       
    
    end
    
    
    filename_raw = natfnames{timeframe_idx * 2 - 1};
    z_size = length(im_frame(1, 1, :));
    
    im_frame = uint8(im_frame);
    for k = 1:z_size
        input = im_frame(:, :, k);
        imwrite(input, strcat(filename_raw,'_output_SINGLE_BAD.tif') , 'writemode', 'append', 'Compression','none')
    end
    
    
    timeframe_idx = timeframe_idx + 1;    
end
