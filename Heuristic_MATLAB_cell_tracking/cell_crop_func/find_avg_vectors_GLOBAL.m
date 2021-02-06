%% Find average vectors of movement for all neighboring cells
function [avg_vec, all_unit_v, all_dist_to_avg, cells_in_crop, all_dist_to_avg_RAW] = find_avg_vectors_GLOBAL(dist_label_idx_matrix, cur_centroids_scaled, next_centroids_scaled,  neighbor_idx, plot_bool)
cells_in_crop = unique(dist_label_idx_matrix);
cells_in_crop = cells_in_crop(2:end);  % get rid of zero

% (2) Find the vectors for all of the cells in the indexes above
all_unit_v = [];
all_v = [];
for idx_crop = 1:length(cells_in_crop)   % skip 0
    check_cell_dist = cells_in_crop(idx_crop);
    
    cur_center =  cur_centroids_scaled(check_cell_dist, :);
    neighbor_center =  next_centroids_scaled(neighbor_idx(check_cell_dist), :);
    
    vector = abs(cur_center) - abs(neighbor_center);
    all_v = [all_v; vector];
    
    %unit_v = vector/norm(vector);
    %all_unit_v = [all_unit_v; unit_v];
    if plot_bool; plot3([0, vector(1)], [0, vector(2)], [0, vector(3)]); end
   
end
avg_vec = mean(all_v, 1);
avg_unit_vec = avg_vec/norm(avg_vec, 1);


if plot_bool
    view(3)
    plot3([0, avg_vec(1)], [0, avg_vec(2)], [0, avg_vec(3)], 'LineWidth', 10, 'color', 'g');
    %plot3([0, avg_unit_vec(1)], [0, avg_unit_vec(2)], [0, avg_unit_vec(3)], 'LineWidth', 10, 'color', 'r');
end


all_dist_to_avg = [];
all_dist_to_avg_RAW = [];
for i = 1:length(all_v)
    dist_to_avg = avg_vec - all_v((i), :);
    all_dist_to_avg = [all_dist_to_avg; norm(dist_to_avg)];
    all_dist_to_avg_RAW = [all_dist_to_avg_RAW; dist_to_avg];
end


end