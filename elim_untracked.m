function [matrix_timeseries_cleaned] = elim_untracked(matrix_timeseries, num_frames_exclude, foldername, natfnames, crop_size, z_size, thresh_size, first_slice, last_slice)

%% Find and delete cells that only exist on "x" frame

%% Find number of non-empty in this current row:
num_non_empty =sum(~cellfun(@isempty,matrix_timeseries),2);

%% Identify remaining unassociated cells and add them to the cell list with NEW numbering (at the end of the list)
matrix_timeseries_cleaned = matrix_timeseries;
disp('checking single cells')
all_s = cell(0);
timeframe_idx = 2;  %% Tiger added - June 6th - skip first timeframe
count = 0;
for sorted_idx = 3:2:length(natfnames) - 2
    
    fileNum = sorted_idx;
    %[all_s, frame_1, truth_1] = load_data_into_struct(foldername, natfnames, fileNum, all_s, thresh_size, first_slice, last_slice);
    
    %im_frame = zeros(size(frame_1));
    for cell_idx = 1:length(matrix_timeseries(:, 1))
        if isempty(matrix_timeseries{cell_idx, timeframe_idx})
            continue;
        end
        cur_cell = matrix_timeseries{cell_idx, timeframe_idx};
        next_cell = matrix_timeseries{cell_idx, timeframe_idx + 1};
        
        voxels = cur_cell.voxelIdxList;
        

        % If next timeseries is empty (AND timeframe_idx == 1 OR previous timeframe is also empty), means that cell was terminated early
        % so plot the output for the next one
        if num_non_empty(cell_idx) <= num_frames_exclude
            frame_1_centroid = cur_cell.centroid;
%             im_size = size(frame_1);
%             height = im_size(1);  width = im_size(2); depth = im_size(3);
%             y = round(frame_1_centroid(1)); x = round(frame_1_centroid(2)); z = round(frame_1_centroid(3));
%             [crop_frame_1, box_x_min ,box_x_max, box_y_min, box_y_max, box_z_min, box_z_max] = crop_around_centroid(frame_1, y, x, z, crop_size, z_size, height, width, depth);
%             
%             blank_truth = zeros(size(truth_1));
%             blank_truth(x, y, z) = 1;
%             crop_blank_truth_1 = crop_around_centroid(blank_truth, y, x, z, crop_size, z_size, height, width, depth);
%             crop_blank_truth_1 = imdilate(crop_blank_truth_1, strel('sphere', 3));
%             
%             mip_1 = max(crop_frame_1, [], 3);
%             figure(1); imshow(mip_1);
%             
%             mip_center_1 = max(crop_blank_truth_1, [], 3);
%             magenta = cat(3, ones(size(mip_1)), zeros(size(mip_1)), ones(size(mip_1)));
%             hold on;
%             h = imshow(magenta);
%             hold off;
%             set(h, 'AlphaData', mip_center_1)
%             title(strcat('x: ', num2str(x), ' y: ', num2str(y), 'z: ', num2str(z)));
            %pause
            
            close all;
            
            %%%im_frame(voxels) = 1;
            disp('yep')
            count = count + 1;
            disp(num2str(count));
            
            matrix_timeseries_cleaned{cell_idx, timeframe_idx} = [];
        end
       
    
    end
    
    
    %filename_raw = natfnames{timeframe_idx * 2 - 1};
    %z_size = length(im_frame(1, 1, :));
    
    %im_frame = uint8(im_frame);
    %for k = 1:z_size
    %    input = im_frame(:, :, k);
    %    imwrite(input, strcat(filename_raw,'_output_SINGLE_BAD.tif') , 'writemode', 'append', 'Compression','none')
    %end
    
    
    timeframe_idx = timeframe_idx + 1;    
end
