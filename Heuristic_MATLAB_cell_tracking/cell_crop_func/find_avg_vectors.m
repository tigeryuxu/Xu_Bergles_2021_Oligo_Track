%% Find average vectors of movement for all neighboring cells
function [avg_vec, all_unit_v, all_dist_to_avg, cells_in_crop] = find_avg_vectors(dist_label_idx_matrix, cur_timeseries, frame_1, crop_size, z_size, cur_centroids, cur_centroids_scaled, next_centroids_scaled, check_neighbor, neighbor_idx, plot_bool, skip)

crop_size = round(crop_size);
z_size = round(z_size);


%% (1) get crop to see what cells are in dist_label_idx_matrix crop
frame_1_centroid = cur_centroids(check_neighbor, :);
% get frame 1 crop
y = round(frame_1_centroid(1)); x = round(frame_1_centroid(2)); z = round(frame_1_centroid(3));
im_size = size(frame_1);
height = im_size(1);  width = im_size(2); depth = im_size(3);
[crop_dist_labels] = crop_around_centroid(dist_label_idx_matrix, y, x, z, crop_size, z_size, height, width, depth);

cells_in_crop = unique(crop_dist_labels);
cells_in_crop = cells_in_crop(2:end);  % get rid of zero


%% skips this if less than 5 cells to get vectors from
if skip && length(cells_in_crop) <= 5
    avg_vec = []; all_unit_v = []; all_dist_to_avg = [];
    return;
end

% (2) Find the vectors for all of the cells in the indexes above
all_unit_v = [];
all_v = [];
for idx_crop = 1:length(cells_in_crop)   % skip 0
    check_cell_dist = cells_in_crop(idx_crop);
    
    cur_center =  cur_centroids_scaled(check_cell_dist, :);
    neighbor_center =  next_centroids_scaled(neighbor_idx(check_cell_dist), :);
    
    vector = cur_center - neighbor_center;
    all_v = [all_v; vector];
    
    %unit_v = vector/norm(vector);
    %all_unit_v = [all_unit_v; unit_v];
    if plot_bool; plot3([0, vector(1)], [0, vector(2)], [0, vector(3)]); end
   
end
avg_vec = mean(all_v, 1);
%avg_unit_vec = avg_vec/norm(avg_vec, 1);


if plot_bool
    view(3)
    plot3([0, avg_vec(1)], [0, avg_vec(2)], [0, avg_vec(3)], 'LineWidth', 4, 'color', 'g');
    %plot3([0, avg_unit_vec(1)], [0, avg_unit_vec(2)], [0, avg_unit_vec(3)], 'LineWidth', 10, 'color', 'r');
end


all_dist_to_avg = [];
for i = 1:length(all_v)
    dist_to_avg = avg_vec - all_v((i), :);
    all_dist_to_avg = [all_dist_to_avg; norm(dist_to_avg)];
end


end