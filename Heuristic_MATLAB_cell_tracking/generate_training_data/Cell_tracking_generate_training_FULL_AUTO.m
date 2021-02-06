%% PROBLEMS:
% (1) the segmentation from PYTORCH might not be the most accurate...
%           so if something is missing on the next frame, might just be
%           because it's not there due to segmentation error...
%           we want it to pick out what's there REGARDLESS of what
%           is/isn't...
% (2) crop around centroid with pads
% (3) should I regenerate the segmentations???
% (4) should it be trained to always just put down a dot of the same
% (5) ***also never gets to see very small cells???

% size??? ==> hausdorf loss paper???

opengl hardware;
close all;

addpath('../IO_func/')
addpath('../cell_crop_func/')


cur_dir = pwd;
addpath(strcat(cur_dir))  % adds path to functions
cd(cur_dir);

%% Initialize
foldername = uigetdir();   % get directory

%% Run Analysis
cd(foldername);   % switch directories
nameCat = '*.tif';
fnames = dir(nameCat);

trialNames = {fnames.name};
numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently

cd(cur_dir);
natfnames=natsort(trialNames);



%% Also get "csv"s
cd(foldername);   % switch directories
nameCat_csv = '*.csv*';
fnames_csv = dir(nameCat_csv);

trialNames_csv = {fnames_csv.name};
numfids_csv = length(trialNames_csv);   %%% divided by 5 b/c 5 files per pack currently

cd(cur_dir);
natfnames_csv =natsort(trialNames_csv);



%% Also get other files for timeseries later

% get RAW
cd(foldername);
cd('./single channel/');   % switch directories
nameCat = '*.tif';
fnames = dir(nameCat);

trialNames = {fnames.name};
numfids_timeseries = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently

cd(cur_dir);
natfnames_timeseries_RAW =natsort(trialNames);


% get seg
cd(foldername);
cd('./a_training_data_GENERATE_FULL_AUTO_output_PYTORCH/');   % switch directories
nameCat = '*.tif';
fnames = dir(nameCat);

trialNames = {fnames.name};
numfids_timeseries = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently

cd(cur_dir);
natfnames_timeseries_SEG =natsort(trialNames);




for fileNum = 1:length(natfnames)
    
    cd(foldername);
    filename_raw = natfnames{fileNum};
    [red_3D, xyres] = load_3D_gray(filename_raw, natfnames);
    z_scale = 1/3;   % is in pixel/um axial depth
    x_scale = xyres;  % is in pixel/um
    y_scale = xyres;
    
    if isempty(xyres)
        x_scale = 1/0.8302662;
        y_scale = 1/0.8302662;
    end
    
    im_size = size(red_3D);
    im_y_size = im_size(2);
    im_x_size = im_size(1);
    im_z_size = im_size(3);
    
    
    %% LOAD ENTIRE SERIES OF RAW AND SEGMENTATION IMAGES
    % find names that contain parts of the filename_raw
    partial_name = filename_raw;
    partial_name = strsplit(partial_name, 'T=0');
    partial_name = partial_name{1};
    
    matching_idx = [];
    for name_idx = 1:length(natfnames_timeseries_RAW)
        cur_name = natfnames_timeseries_RAW{name_idx};
        if contains(cur_name, partial_name)
            matching_idx = [matching_idx, name_idx];
        end
    end
    
    % (1) load raw
    cd('./single channel/');   % switch directories
    
    timeseries_RAW = zeros([im_x_size, im_y_size, im_z_size, length(matching_idx)]);
    for m_idx = 1:length(matching_idx)
        im_name = natfnames_timeseries_RAW{matching_idx(m_idx)};
        im = load_3D_gray(im_name, natfnames_timeseries_RAW);
        timeseries_RAW(:, :, :, m_idx) = im;
    end
    timeseries_RAW = im2uint8(timeseries_RAW);
    
    
    % (2) load segmentation images
    cd(foldername);
    cd('./a_training_data_GENERATE_FULL_AUTO_output_PYTORCH/');   % switch directories
    
    timeseries_SEG =  zeros([im_x_size, im_y_size, im_z_size, length(matching_idx)]);
    all_cc = cell(0);
    for m_idx = 1:length(matching_idx)
        im_name = natfnames_timeseries_SEG{matching_idx(m_idx)};
        im = load_3D_gray(im_name, natfnames_timeseries_SEG);
        timeseries_SEG(:, :, :, m_idx) = im;
        
        cc = bwconncomp(imbinarize(im));
        all_cc{end + 1} = cc.PixelIdxList;
    end
    timeseries_SEG = im2uint8(timeseries_SEG);
    
    cd(foldername);
    
    
    
    %% Load csv as well
    
    filename_raw_csv = natfnames_csv{fileNum};
    syGlass10x = readtable(filename_raw_csv);
    
    %% START
    save_name = filename_raw;
    save_name = split(save_name, '.tif');
    save_name = strjoin(save_name(1:end - 1));
    
    frame = syGlass10x.FRAME;
    all_Z = syGlass10x.Z;
    all_X = syGlass10x.Y;
    all_Y = syGlass10x.X;
    
    %% Scale x and y
    all_X = all_X * x_scale;
    all_Y = all_Y * y_scale;
    
    %% Normalize to first val 0 indexing
    middle_val = im_x_size ./ 2;
    all_X = round(all_X + middle_val);
    
    middle_val = im_y_size ./ 2;
    all_Y = round(all_Y + middle_val);
    
    %% Scale Z
    all_Z = all_Z * z_scale;
    middle_val = im_z_size ./ 2;
    all_Z = round(all_Z + middle_val);
    
    %% Tiger - add row of index
    indices = 1:length(frame);
    
    
    together = [frame, all_X, all_Y, all_Z, indices'];
    [~,idx] = sort(together(:,1)); % sort just the first column
    sortedmat = together(idx,:);   % sort the whole matrix using the sort indices
    
    cur_idx = 0;
    im_size = [im_x_size, im_y_size, im_z_size];
    blank_im = zeros(im_size);
    for i = 1:length(sortedmat)
        
        cur_frame = sortedmat(i);
        
        %% ***stop if next frame is the last frame
        if cur_idx + 1 == m_idx
            break;
            
            
        elseif cur_idx == cur_frame
            % plot
            x = sortedmat(i, 2);
            if x > im_x_size
                x = im_x_size
            elseif x <= 0
                x = 1;
            end
            
            y = sortedmat(i, 3);
            if y > im_y_size
                y = im_y_size
            elseif y <= 0
                y = 1;
            end
            z = sortedmat(i, 4);
            if z > im_z_size
                z = im_z_size
            elseif z <= 0
                z = 1;
            end
            
            lin_ind = sub2ind(size(im), x, y, z);
            
            
            
            %% also get the centroid for the next frame so can get match cc
            cur_cell_idx = sortedmat(i, 5);
            next_cell_idx = cur_cell_idx + 1;
            j = find(sortedmat(:, 5) == next_cell_idx);
            
            
            % plot
            x_next = sortedmat(j, 2);
            if x_next > im_x_size
                x_next = im_x_size
            elseif x_next <= 0
                x_next = 1;
            end
            
            y_next = sortedmat(j, 3);
            if y_next > im_y_size
                y_next = im_y_size
            elseif y_next <= 0
                y_next = 1;
            end
            z_next = sortedmat(j, 4);
            if z_next > im_z_size
                z_next = im_z_size
            elseif z_next <= 0
                z_next = 1;
            end
            
            lin_ind_next = sub2ind(size(im), x_next, y_next, z_next);
            
            %% ***DONT ALLOW DOUBLES
            
            %% ***by doing watershed???
            
            %% CROP cur timeseries AND next timeseries
            blank_im(lin_ind) = 1;
            
            raw_cur = im2double(timeseries_RAW(:, :, :, cur_idx + 1));
            raw_next = im2double(timeseries_RAW(:, :, :, cur_idx + 2));
            
            
            cc_cur = all_cc(cur_idx + 1);
            cc_next = all_cc(cur_idx + 2);
            
            %% Find out which cc matches with current lin_ind and lin_ind_next
            lower = 2; upper = 2;
            
            [expanded, lin_ind_expand] =  expand_coord_to_neighborhood([x, y, z], lower, upper, im_size);
            
            match_1 = 0;
            seg_cur = zeros(size(im));
            for cc_idx = 1:length(cc_cur{1})
                cur = cc_cur{1}{cc_idx};
                if ~isempty(find(ismember(cur, lin_ind_expand)))
                    disp('matched')
                    matched_cur = cur;
                    seg_cur(matched_cur) = 1;
                    match_1 = 1;
                    break;
                end
            end
            
            
            %% ALSO set match_2 == 1 if the cell dies in next timeseries:
            if ~isempty(j)   %% if the next cell is DEAD, then still allow tracking!
                [expanded, lin_ind_next_exapnd] =  expand_coord_to_neighborhood([x_next, y_next, z_next], lower, upper, im_size);
                
                frame_num_check = sortedmat(j, 1);
                if frame_num_check ~= cur_frame + 1
                    match_2 = 1;
                    seg_next = zeros(size(im));
                    disp('DEAD CELL');
                else
                    % do for next as well
                    match_2 = 0;
                    seg_next = zeros(size(im));
                    for cc_idx = 1:length(cc_next{1})
                        cur = cc_next{1}{cc_idx};
                        if ~isempty(find(ismember(cur, lin_ind_next_exapnd)))
                            disp('matched')
                            matched_cur_next = cur;
                            seg_next(matched_cur_next) = 1;
                            match_2 = 1;
                            break;
                        end
                    end
                end
                
            else
                
                match_2 = 1;   %% if the next cell is DEAD, then still allow tracking!
                
            end
            
            
            %% ONLY IF BOTH EXIST:
            if match_1 == 1 && match_2 == 1
                
                % (1) crop the raw
                crop_size = 80;
                z_size = 32;
                crop_raw_cur = crop_around_centroid_with_pads(raw_cur, y, x, z, crop_size, z_size, im_x_size, im_y_size, im_z_size);
                crop_seg_cur = crop_around_centroid_with_pads(seg_cur, y, x, z, crop_size, z_size, im_x_size, im_y_size, im_z_size);
                
                crop_raw_next = crop_around_centroid_with_pads(raw_next, y, x, z, crop_size, z_size, im_x_size, im_y_size, im_z_size);
                crop_seg_next = crop_around_centroid_with_pads(seg_next, y, x, z, crop_size, z_size, im_x_size, im_y_size, im_z_size);
                
                
                %             else
                %                 % (1) crop the raw
                %                 if match_1 == 0
                %                     seg_cur = zeros(size(im));
                %                     seg_cur(lin_ind) = 1;
                %
                %                     %seg_cur = imdilate(seg_cur, strel('sphere', 5));
                %                 end
                %
                %                 if match_2 == 0
                %                     seg_next = zeros(size(im));
                %                     seg_next(lin_ind_next) = 1;
                %
                %                     %seg_next = imdilate(seg_next, strel('sphere', 5));
                %                 end
                %
                %                 crop_size = 80;
                %                 z_size = 32;
                %                 crop_raw_cur = crop_around_centroid_with_pads(raw_cur, y, x, z, crop_size, z_size, im_x_size, im_y_size, im_z_size);
                %                 crop_seg_cur = crop_around_centroid_with_pads(seg_cur, y, x, z, crop_size, z_size, im_x_size, im_y_size, im_z_size);
                %                 crop_seg_cur = imdilate(crop_seg_cur, strel('sphere', 5));
                %
                %
                %
                %                 crop_raw_next = crop_around_centroid_with_pads(raw_next, y, x, z, crop_size, z_size, im_x_size, im_y_size, im_z_size);
                %                 crop_seg_next = crop_around_centroid_with_pads(seg_next, y, x, z, crop_size, z_size, im_x_size, im_y_size, im_z_size);
                %                 crop_seg_next = imdilate(crop_seg_next, strel('sphere', 5));
                %
                %             end
                
                %% DEBUG:
                %                 mip1 = plot_max(crop_raw_cur);
                %                 mip2 = plot_max(crop_seg_cur);
                %
                %                 mip3 = plot_max(crop_raw_next);
                %                 mip4 = plot_max(crop_seg_next);
                
                %% SAVE:
                mip1 = max(crop_raw_cur, [], 3);
                mip2 = max(crop_seg_cur, [], 3);
                
                mip3 = max(crop_raw_next, [], 3);
                mip4 = max(crop_seg_next, [], 3);
                cd(foldername)
                cd('./mip quality check/')
                
                if exist(strcat('./', save_name), 'dir') == 0
                    mkdir(save_name)
                end
                cd(save_name);
                
                imwrite(mip1, strcat(save_name, '_frame_' , num2str(cur_idx), '_', num2str(i) , '_mip1.tif') , 'writemode', 'append', 'Compression','none')
                %imwrite(mip2, strcat(save_name, '_frame_' , num2str(cur_idx), '_', num2str(i) , '_mip2.tif') , 'writemode', 'append', 'Compression','none')
                imwrite(mip3, strcat(save_name, '_frame_' , num2str(cur_idx), '_', num2str(i), '_mip4.tif') , 'writemode', 'append', 'Compression','none')
                %imwrite(mip4, strcat(save_name, '_frame_' , num2str(cur_idx), '_', num2str(i) ,'_mip5.tif') , 'writemode', 'append', 'Compression','none')
                
                %figure; imshow(mip_RGB)
                
                %% ALSO GET THE ACTUAL FULL SEGMENTATION TO GUIDE THE SEED?
                seg_cur = im2double(timeseries_SEG(:, :, :, cur_idx + 1));
                seg_next = im2double(timeseries_SEG(:, :, :, cur_idx + 2));
                
                crop_seg_cur_FULL = crop_around_centroid_with_pads(seg_cur, y, x, z, crop_size, z_size, im_x_size, im_y_size, im_z_size);
                crop_seg_next_FULL = crop_around_centroid_with_pads(seg_next, y, x, z, crop_size, z_size, im_x_size, im_y_size, im_z_size);
                %
                mip5 = max(crop_seg_cur_FULL, [], 3);
                mip6 = max(crop_seg_next_FULL, [], 3);
                
                
                mip5_RGB = cat(3, mip5, mip2, zeros(size(mip2)));
                mip6_RGB = cat(3, mip6, mip4, zeros(size(mip2)));
                
                imwrite(mip5_RGB, strcat(save_name, '_frame_' , num2str(cur_idx), '_', num2str(i), '_mip3.tif') , 'writemode', 'append', 'Compression','none')
                imwrite(mip6_RGB, strcat(save_name, '_frame_' , num2str(cur_idx), '_', num2str(i) ,'_mip6.tif') , 'writemode', 'append', 'Compression','none')
                
                
                %% SAVE:
                cd(foldername)
                cd('./Training cell track full auto');
                if exist(strcat('./', save_name), 'dir') == 0
                    mkdir(save_name)
                end
                cd(save_name);
                
                
                %figure(1); volshow(crop_input);
                for k = 1:z_size
                    input = crop_raw_cur(:, :, k);
                    imwrite(input,  strcat(save_name, '_frame_' , num2str(cur_idx), '_', num2str(i) , '_crop_input_cur.tif'), 'writemode', 'append', 'Compression','none')
                    % save cropped seeds
                    
                    input = crop_seg_cur(:, :, k);
                    imwrite(input,  strcat(save_name, '_frame_' , num2str(cur_idx), '_', num2str(i) , '_crop_input_cur_seed.tif'), 'writemode', 'append', 'Compression','none')
                    
                    input = crop_seg_cur_FULL(:, :, k);
                    imwrite(input,  strcat(save_name, '_frame_' , num2str(cur_idx), '_', num2str(i) , '_crop_input_cur_seg_FULL.tif'), 'writemode', 'append', 'Compression','none')
                    
                    input = crop_raw_next(:, :, k);
                    imwrite(input,  strcat(save_name, '_frame_' , num2str(cur_idx), '_', num2str(i) , '_crop_input_next.tif'), 'writemode', 'append', 'Compression','none')
                    
                    input = crop_seg_next_FULL(:, :, k);
                    imwrite(input,  strcat(save_name, '_frame_' , num2str(cur_idx), '_', num2str(i) , '_crop_input_next_seg_FULL.tif'), 'writemode', 'append', 'Compression','none')
                    
                    input = crop_seg_next(:, :, k);
                    imwrite(input,  strcat(save_name, '_frame_' , num2str(cur_idx), '_', num2str(i) , '_crop_truth.tif'), 'writemode', 'append', 'Compression','none')
                    
                    
                    
                end
                
            end
            
        else
            cur_idx = cur_idx + 1;
        end
    end
    
end