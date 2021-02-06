function [crop_frame_1, crop_frame_2, crop_truth_1, crop_truth_2, mip_1, mip_2, crop_blank_truth_1, crop_blank_truth_2] = crop_centroids(cur_centroids, next_centroids, frame_1, frame_2, truth_1, truth_2, check_neighbor, neighbor_idx, crop_size, z_size)
    
    crop_size = round(crop_size);
    z_size = round(z_size);

    frame_1_centroid = cur_centroids(check_neighbor, :);
    frame_2_centroid = next_centroids(neighbor_idx(check_neighbor), :);

    % get frame 1 crop
    y = round(frame_1_centroid(1)); x = round(frame_1_centroid(2)); z = round(frame_1_centroid(3));
    im_size = size(frame_1);
    height = im_size(1);  width = im_size(2); depth = im_size(3);
    crop_frame_1 = crop_around_centroid(frame_1, y, x, z, crop_size, z_size, height, width, depth);
    crop_truth_1 = crop_around_centroid(truth_1, y, x, z, crop_size, z_size, height, width, depth);
 
    % get a truth with ONLY the current cell of interest
    blank_truth = zeros(size(truth_1));
    blank_truth(x, y, z) = 1;
    %blank_truth = imdilate(blank_truth, strel('sphere', 3));
    crop_blank_truth_1 = crop_around_centroid(blank_truth, y, x, z, crop_size, z_size, height, width, depth);
    crop_blank_truth_1 = imdilate(crop_blank_truth_1, strel('sphere', 2));
    
    mip_1 = max(crop_frame_1, [], 3);
    %subplot(1, 2, 1); imshow(mip_1);

    % get frame 2 crop
    y = round(frame_2_centroid(1)); x = round(frame_2_centroid(2)); z = round(frame_2_centroid(3));
    im_size = size(frame_2);
    height = im_size(1);  width = im_size(2); depth = im_size(3);
    crop_frame_2 = crop_around_centroid(frame_2, y, x, z, crop_size, z_size, height, width, depth);
    crop_truth_2 = crop_around_centroid(truth_2, y, x, z, crop_size, z_size, height, width, depth);

    % get a truth with ONLY the current cell of interest
    blank_truth = zeros(size(truth_2));
    blank_truth(x, y, z) = 1;
    %blank_truth = imdilate(blank_truth, strel('sphere', 3));
    crop_blank_truth_2 = crop_around_centroid(blank_truth, y, x, z, crop_size, z_size, height, width, depth);
    crop_blank_truth_2 = imdilate(crop_blank_truth_2, strel('sphere', 2));
    
    mip_2 = max(crop_frame_2, [], 3);
    %subplot(1, 2, 2); imshow(mip_2);
end