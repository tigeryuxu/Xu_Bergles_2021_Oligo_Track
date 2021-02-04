function [idx_double_counted] = Bergles_manual_correct_double_count(frame_num, frame_1, truth_1, matrix_timeseries, crop_size, z_size)

% Allows user to manually correct double counted cells

idx_double_counted = [];
% for frame_num = length(matrix_timeseries(1, :)):-1:2   % loops back to the 2nd to last frame
%     %disp(strcat('checking frame ', {' '}, num2str(frame_num), {' '}, 'for duplicates'));
%
already_checked = [];
for cell_num = 1:length(matrix_timeseries(:, frame_num))
    cur_cell = matrix_timeseries{cell_num, frame_num};
    if isempty(cur_cell); continue; end
    
    if ismember(cell_num, already_checked); continue; end
    
    num_duplicates = 0;
    % goes through and checks the current cells with all other cells in
    % the list to see if there are duplicates
    for check_cell_idx = 1:length(matrix_timeseries(:, frame_num))
        if check_cell_idx == cell_num
            continue;
        end
        
        if ismember(check_cell_idx, already_checked); continue; end
        
        
        check_cell = matrix_timeseries{check_cell_idx, frame_num};
        if isempty(check_cell); continue; end
        
        
        same = ismember(cur_cell.voxelIdxList, check_cell.voxelIdxList);
        if ~isempty(find(same))
            num_duplicates = num_duplicates + 1;
            already_checked = [already_checked; check_cell_idx];
            already_checked = [already_checked; cell_num];
            idx_double_counted = [idx_double_counted; [frame_num, check_cell_idx, cell_num]];
        end
        
    end
    
end

