%% read csv for comparison

addpath('../cell_crop_func')
%T = readtable('680_syGlass_10x.csv');
%T = readtable('650_syGlass_10x.csv');
%T = readtable('output_650.csv');
%T = readtable('output_680.csv');

%T = readtable('a1901128-r670_syGlass_20x.csv');

%T = readtable('MOBPF1_013018_cuprBZA_10x.tif - T=0_610_syGlass_10x.csv');

%T = readtable('MOBPM_190226w_2_10x_Reg2.tif - T=0_a1902262-r720_syGlass_20x.csv');

%T = readtable('output.csv');

%T = readtable('syglassCheck.csv');


T = readtable('MOBPF_190627w_5_syglassCorrectedTracks.csv');


matrix_timeseries = cell(5000, max(T.FRAME) + 1);

for i = 1:length(T.SERIES )
%     
    cell_num = T.SERIES(i) + 1;
    frame_num = T.FRAME(i) + 1;
%    
%   cell_num = T.Var1(i) + 1;
%   frame_num = T.Var2(i) + 1;
   
   centroid = [];
   voxelIdxList = [];
   
   cell = cell_class(voxelIdxList, centroid, cell_num);
   
   matrix_timeseries{cell_num, frame_num} = cell;
    
end


%% Plot number of new cells and number of old cells at each timepoint
new_cells_per_frame = zeros(1, length(matrix_timeseries(1, :)));
terminated_cells_per_frame = zeros(1, length(matrix_timeseries(1, :)));
num_total_cells_per_frame = zeros(1, length(matrix_timeseries(1, :)));
for timeframe_idx = 1:length(matrix_timeseries(1, :))
    for cell_idx = 1:length(matrix_timeseries(:, 1))

        if isempty(matrix_timeseries{cell_idx, timeframe_idx})
            continue;
        end
        cur_cell = matrix_timeseries{cell_idx, timeframe_idx};
        
        % new cell if previous frame empty
        if timeframe_idx > 1 && isempty(matrix_timeseries{cell_idx, timeframe_idx - 1})
            new_cells_per_frame(timeframe_idx) =  new_cells_per_frame(timeframe_idx) + 1;
        end
        
        % terminated cells if next frame empty
        if timeframe_idx + 1 < length(matrix_timeseries(1, :)) && isempty(matrix_timeseries{cell_idx, timeframe_idx + 1})
            terminated_cells_per_frame(timeframe_idx + 1) =  terminated_cells_per_frame(timeframe_idx + 1) + 1;
        end
        
        % number of totl cells per frame
        num_total_cells_per_frame(timeframe_idx) = num_total_cells_per_frame(timeframe_idx) + 1;
        
    end
   
end

figure; bar(new_cells_per_frame); title('New cells per frame');
xlabel('frame number'); ylabel('number of new cells');

figure; bar(terminated_cells_per_frame); title('terminated cells per frame');
xlabel('frame number'); ylabel('number of terminated cells');

figure; bar(num_total_cells_per_frame); title('num TOTAL cells per frame');
xlabel('frame number'); ylabel('num TOTAL cells');



%% Save image with only the new cells on this frame
% frame_num = 8;
% timeframe_idx = frame_num;
% 
% % can be modified:
% im_frame = zeros([1024, 1024, 185]);
% %[all_s, frame_1, truth_1] = load_data_into_struct(foldername, natfnames, fileNum, all_s, thresh_size, first_slice, last_slice);
% for cell_idx = 1:length(matrix_timeseries(:, 1))
%     
%     if isempty(matrix_timeseries{cell_idx, timeframe_idx})
%         continue;
%     end
%     
%     % new cell if previous frame empty
%     if timeframe_idx > 1 && isempty(matrix_timeseries{cell_idx, timeframe_idx - 1})
%         disp('in')
%         cur_cell = matrix_timeseries{cell_idx, timeframe_idx};
%         
%         voxels = cur_cell.voxelIdxList;
%         %cell_number = cur_cell.cell_number;
%         
%         im_frame(voxels) = 1;
%     end
% end
% 
% % save one frame
% filename_raw = 'only_new_cells';
% z_size = length(im_frame(1, 1, :));
% 
% im_frame = uint8(im_frame);
% for k = 1:z_size
%     input = im_frame(:, :, k);
%     imwrite(input, strcat(filename_raw,'_current_frame.tif') , 'writemode', 'append', 'Compression','none')
% end



