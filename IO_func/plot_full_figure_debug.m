function [] = plot_full_figure_debug(frame_1, frame_2, truth_1, truth_2, crop_frame_1, crop_frame_2, crop_truth_1, crop_truth_2, D, check_neighbor, neighbor_idx, matrix_timeseries, cur_timeseries, next_timeseries, timeframe_idx, crop_size, z_size,cur_centroids, crop_blank_truth_1, crop_blank_truth_2, next_centroids)

mip_1 = max(crop_frame_1, [], 3);
opt = 0;

im_size = size(frame_1);
height = im_size(1);  width = im_size(2); depth = im_size(3);

frame_1_centroid = cur_centroids(check_neighbor, :);
% get frame 1 crop
y = round(frame_1_centroid(1)); x = round(frame_1_centroid(2)); z = round(frame_1_centroid(3));

%% Parse frame 2
frame_2_centroid = next_centroids(neighbor_idx(check_neighbor), :);

%% Do not center
im_size = size(frame_2);
height = im_size(1);  width = im_size(2); depth = im_size(3);
[crop_frame_2, x_min, x_max, y_min, y_max, z_min, z_max] = crop_around_centroid(frame_2, y, x, z, crop_size, z_size, height, width, depth);
crop_truth_2 = crop_around_centroid(truth_2, y, x, z, crop_size, z_size, height, width, depth);

%% get a truth with ONLY the current cell of interest
y_2 = round(frame_2_centroid(1)); x_2 = round(frame_2_centroid(2)); z_2 = round(frame_2_centroid(3));
blank_truth = zeros(size(truth_2));
blank_truth(x_2, y_2, z_2) = 1;
%blank_truth = imdilate(blank_truth, strel('sphere', 3));
crop_blank_truth_2 = crop_around_centroid(blank_truth, y, x, z, crop_size, z_size, height, width, depth);
crop_blank_truth_2 = imdilate(crop_blank_truth_2, strel('sphere', 2));




mip_2 = max(crop_frame_2, [], 3);
%subplot(1, 2, 2); imshow(mip_2);


%% accuracy metrics
dist = D(check_neighbor);
ssim_val = ssim(crop_frame_1, crop_frame_2);
mae_val = meanAbsoluteError(crop_frame_1, crop_frame_2);
psnr_val = psnr(crop_frame_1, crop_frame_2);


%% get a truth with everything LEADING UP TO the current cell of interest (i.e. the tail)
blank_truth = zeros(size(truth_2));
for back_track_idx = 1:length(matrix_timeseries(check_neighbor, 1:timeframe_idx)) % is -1 because exclude current timeframe
    if ~isempty(matrix_timeseries{check_neighbor, back_track_idx})
        prev_x = round(matrix_timeseries{check_neighbor, back_track_idx}.centroid(1));
        prev_y = round(matrix_timeseries{check_neighbor, back_track_idx}.centroid(2));
        prev_z = round(matrix_timeseries{check_neighbor, back_track_idx}.centroid(3));
        blank_truth(prev_y, prev_x, prev_z) = 1;
    end
end
crop_blank_truth_2_PREV = crop_around_centroid(blank_truth, y, x, z, crop_size, z_size, height, width, depth);
crop_blank_truth_2_PREV = imdilate(crop_blank_truth_2_PREV, strel('sphere', 1));








crop_truth_1(crop_blank_truth_1 == 1) = 0;
crop_truth_2(crop_blank_truth_2 == 1) = 0;

if opt == 8
    RGB_1 = cat(4, crop_frame_1, crop_blank_truth_1, crop_blank_truth_1);
    RGB_2 = cat(4, crop_frame_2, crop_blank_truth_2, crop_blank_truth_2);
else
    RGB_1 = cat(4, crop_truth_1, crop_frame_1, crop_blank_truth_1);
    RGB_2 = cat(4, crop_truth_2, crop_frame_2, crop_blank_truth_2);
end

%f = figure('units','normalized','outerposition',[0 0 1 1])
f = figure();
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
p = uipanel();

top_left = uipanel('Parent',p,  'Position',[0.05 0.5 .40 .50]);
top_right = uipanel('Parent',p, 'Position', [.55 0.5 .40 .50]);
bottom_left = uipanel('Parent',p,  'Position',[0.05 0 .40 .50]);
bottom_right = uipanel('Parent',p, 'Position', [.55 0 .40 .50]);

s1 = sliceViewer(RGB_1, 'parent', top_left);
s2 = sliceViewer(RGB_2, 'parent', top_right);

%% (1) Plot bottom LEFT graph - plot max
if opt == 'adjust'
    mip_1 = adapthisteq(mip_1);
end
ax = axes('parent', bottom_left);
imshow(mip_1);
%image(ax, im2uint8(mip_1));
colormap('gray'); axis off

%% Add overlay of tracked cells with diff colors
% (1) Create rainbow thing and get crop
% (2) loop through and use the create overlay to make for each color type
% Recreate the output DAPI for each frame with cell number for each (create rainbow image)
im_frame = zeros(size(frame_1));
im_frame_2 = zeros(size(frame_2));
for cell_idx = 1:length(matrix_timeseries(:, 1))
    
    % only add color to cells that are matched on the next frame
    if isempty(matrix_timeseries{cell_idx, timeframe_idx + 1})
        continue;
    end
 
    % add the 2nd frame color at the same time
    cur_cell = matrix_timeseries{cell_idx, timeframe_idx + 1};
    voxels = cur_cell.voxelIdxList;
    im_frame_2(voxels) = cell_idx;
    
    
    if ~isempty( matrix_timeseries{cell_idx, timeframe_idx})
        cur_cell = matrix_timeseries{cell_idx, timeframe_idx};
        voxels = cur_cell.voxelIdxList;
        im_frame(voxels) = cell_idx;
    end
end

labels_crop_truth_1 = crop_around_centroid(im_frame, y, x, z, crop_size, z_size, height, width, depth);

unique_cells = unique(labels_crop_truth_1);
list_random_colors = rand([length(matrix_timeseries), 3]);
for color_idx = 1:length(unique_cells)
    cell_idx = unique_cells(color_idx);
    if cell_idx == 0; continue; end
    crop_only_cur_color = labels_crop_truth_1;
    crop_only_cur_color(crop_only_cur_color ~= cell_idx) = 0;
    mip_center_1 = max(crop_only_cur_color, [], 3);
    
    %red_val = rand(1) * ones(size(mip_center_1));
    cur_color = list_random_colors(cell_idx, :);
    green_val = cur_color(2) * ones(size(mip_center_1));
    blue_val = cur_color(3) * ones(size(mip_center_1));
    
    %color_mat = cat(3, zeros(size(mip_center_1)), green_val, blue_val);
    
    %% MAKE IT A SINGLE COLOR
    color_mat = cat(3, ones(size(mip_center_1)),ones(size(mip_center_1)), zeros(size(mip_center_1)));
    
    hold on;
    h = imshow(color_mat);
    hold off;
    set(h, 'AlphaData', mip_center_1)
end

%% add overlay of current cell
if opt == 8
    disp('hide');
else
    mip_center_1 = max(crop_blank_truth_1, [], 3);
    green = cat(3, zeros(size(mip_1)), ones(size(mip_1)), zeros(size(mip_1)));
    hold on;
    h = imshow(green);
    hold off;
    set(h, 'AlphaData', mip_center_1)
end


title(strcat('ssim: ', num2str(ssim_val), '  dist: ', num2str(dist)))
text(-88, 0, strcat('Frame: ',{' '}, num2str(timeframe_idx), ' of total: ', {' '},num2str(length(neighbor_idx))))

text(-88, 20, strcat('Green dot:'))
text(-88, 30, strcat('location on left frame') )

text(-88, 50, strcat('Magenta dot:'))
text(-88, 60, strcat('location on right frame') )

text(-88, 80, strcat('Blue line:'))
text(-88, 90, strcat('loc on previous frames') )



%% (2) Plot XZ view of right frame + track of line movement???
%% ***MAKE SURE THE SCALING IS ALWAYS ACCURATE

middle = uipanel('Parent',p, 'Position', [.367 0.2 .28 .20]);
% set 0 to things +/- 20 pixels away from the current centroid to
% clear up the XZ projection a bit
crop_frame_2_temp = crop_frame_2;
location = find(crop_blank_truth_1);
[a, b, c] = ind2sub(size(crop_blank_truth_1), location);
x_center = round(mean(a));
pad = 20;
top = x_center + pad; bottom = x_center - pad;
if x_center + pad + 2 > length(crop_frame_2_temp(:, 1, 1))
    top = length(crop_frame_2_temp(:, 1, 1));
elseif x_center - pad - 2 <= 0
    bottom = 1;
end
crop_frame_2_temp(1:bottom, :, :) = 0;
crop_frame_2_temp(top:end, :, :) = 0;

mip_2_XZ = squeeze(max(crop_frame_2_temp, [], 1));
mip_2_XZ = permute(mip_2_XZ, [2 1]);  XZ_size = size(mip_2_XZ);



mip_2_XZ = imresize(mip_2_XZ, [XZ_size(1) * 3, XZ_size(2) * 0.83]);
ax = axes('parent', middle);
imshow(mip_2_XZ);
colormap('gray'); axis off

% Add overlay of previous track
if opt == 8
    disp('hide');
else
    mip_center_1 = squeeze(max(crop_blank_truth_2_PREV, [], 1));
    mip_center_1 = permute(mip_center_1, [2 1]);
    XZ_size = size(mip_center_1);
    mip_center_1 = imresize(mip_center_1, [XZ_size(1) * 3, XZ_size(2) * 0.83]);
    
    % CONNECT TO FORM LINE
    mip_center_1 = imdilate(mip_center_1, strel('disk', 2));
    mip_center_1 = bwskel(imbinarize(mip_center_1));
    mip_center_1 = imdilate(mip_center_1, strel('disk', 1));
    %figure; imshow(mip_center_1);
    
    blue = cat(3, zeros(size(mip_center_1)), zeros(size(mip_center_1)), ones(size(mip_center_1)));
    hold on;
    h = imshow(blue);
    hold off;
    set(h, 'AlphaData', mip_center_1)
end

%% add overlay of where cell is on the ORIGINAL timeframe
if opt == 8
    disp('hide');
else
    mip_center_1 = squeeze(max(crop_blank_truth_1, [], 1));
    mip_center_1 = permute(mip_center_1, [2 1]);
    mip_center_1 = bwskel(imbinarize(mip_center_1));
    
    XZ_size = size(mip_center_1);
    mip_center_1 = imresize(mip_center_1, [XZ_size(1) * 3, XZ_size(2) * 0.83]);
    
    mip_center_1 = imdilate(mip_center_1, strel('disk', 1));
    magenta = cat(3, zeros(size(mip_center_1)), ones(size(mip_center_1)), zeros(size(mip_center_1)));
    hold on;
    h = imshow(magenta);
    hold off;
    set(h, 'AlphaData', mip_center_1)
end



%% add overlay of current cell
if opt == 8
    disp('hide');
else
    mip_center_2 = squeeze(max(crop_blank_truth_2, [], 1));
    mip_center_2 = permute(mip_center_2, [2 1]);
    mip_center_2 = bwskel(imbinarize(mip_center_2));
    XZ_size = size(mip_center_2);
    mip_center_2 = imresize(mip_center_2, [XZ_size(1) * 3, XZ_size(2) * 0.83]);
    
    %mip_center_2 = imero(imbinarize(mip_center_2));
    mip_center_2 = imdilate(mip_center_2, strel('disk', 1));
    
    magenta = cat(3, ones(size(mip_center_2)), zeros(size(mip_center_2)), ones(size(mip_center_2)));
    hold on;
    h = imshow(magenta);
    hold off;
    set(h, 'AlphaData', mip_center_2)
end

title('RIGHT frame: Scaled XZ project + tracking (centered crop +/- 30 px)')



%% (2.2) Plot XZ view of left frame + track of line movement???
%% ***MAKE SURE THE SCALING IS ALWAYS ACCURATE

middle_top = uipanel('Parent',p, 'Position', [.367 0.6 .28 .20]);
% set 0 to things +/- 20 pixels away from the current centroid to
% clear up the XZ projection a bit
crop_frame_1_temp = crop_frame_1;
location = find(crop_blank_truth_1);
[a, b, c] = ind2sub(size(crop_blank_truth_1), location);
x_center = round(mean(a));
pad = 20;
top = x_center + pad; bottom = x_center - pad;
if x_center + pad + 2 > length(crop_frame_1_temp(:, 1, 1))
    top = length(crop_frame_1_temp(:, 1, 1));
elseif x_center - pad - 2 <= 0
    bottom = 1;
end
crop_frame_1_temp(1:bottom, :, :) = 0;
crop_frame_1_temp(top:end, :, :) = 0;

mip_1_XZ = squeeze(max(crop_frame_1_temp, [], 1));
mip_1_XZ = permute(mip_1_XZ, [2 1]);  XZ_size = size(mip_1_XZ);



mip_1_XZ = imresize(mip_1_XZ, [XZ_size(1) * 3, XZ_size(2) * 0.83]);
ax = axes('parent', middle_top);
imshow(mip_1_XZ);
colormap('gray'); axis off

% Add overlay of previous track
if opt == 8
    disp('hide');
else
    mip_center_1 = squeeze(max(crop_blank_truth_2_PREV, [], 1));
    mip_center_1 = permute(mip_center_1, [2 1]);
    XZ_size = size(mip_center_1);
    mip_center_1 = imresize(mip_center_1, [XZ_size(1) * 3, XZ_size(2) * 0.83]);
    
    % CONNECT TO FORM LINE
    mip_center_1 = imdilate(mip_center_1, strel('disk', 2));
    mip_center_1 = bwskel(imbinarize(mip_center_1));
    mip_center_1 = imdilate(mip_center_1, strel('disk', 1));
    %figure; imshow(mip_center_1);
    
    blue = cat(3, zeros(size(mip_center_1)), zeros(size(mip_center_1)), ones(size(mip_center_1)));
    hold on;
    h = imshow(blue);
    hold off;
    set(h, 'AlphaData', mip_center_1)
end

%% add overlay of current cell
if opt == 8
    disp('hide');
else
    mip_center_1 = squeeze(max(crop_blank_truth_1, [], 1));
    mip_center_1 = permute(mip_center_1, [2 1]);
    mip_center_1 = bwskel(imbinarize(mip_center_1));
    
    XZ_size = size(mip_center_1);
    mip_center_1 = imresize(mip_center_1, [XZ_size(1) * 3, XZ_size(2) * 0.83]);
    
    mip_center_1 = imdilate(mip_center_1, strel('disk', 1));
    magenta = cat(3, zeros(size(mip_center_1)), ones(size(mip_center_1)), zeros(size(mip_center_1)));
    hold on;
    h = imshow(magenta);
    hold off;
    set(h, 'AlphaData', mip_center_1)
end

title('LEFT frame: Scaled XZ project + tracking (crop +/- 30 px)')




%% (3) Plot bottom RIGHT graph - plot max
if opt == 'adjust'
    mip_2 = adapthisteq(mip_2);
end
ax = axes('parent', bottom_right);
imshow(mip_2);
colormap('gray'); axis off
%% Add overlay of tracked cells with diff colors
labels_crop_truth_2 = crop_around_centroid(im_frame_2, y_2, x_2, z_2, crop_size, z_size, height, width, depth);

unique_cells = unique(labels_crop_truth_2);
for color_idx = 1:length(unique_cells)
    cell_idx = unique_cells(color_idx);
    if cell_idx == 0; continue; end
    crop_only_cur_color = labels_crop_truth_2;
    crop_only_cur_color(crop_only_cur_color ~= cell_idx) = 0;
    mip_center_2 = max(crop_only_cur_color, [], 3);
    
    %red_val = rand(1) * ones(size(mip_center_2));
    cur_color = list_random_colors(cell_idx, :);
    green_val = cur_color(2) * ones(size(mip_center_2));
    blue_val = cur_color(3) * ones(size(mip_center_2));
    
    
    %color_mat = cat(3, zeros(size(mip_center_1)),green_val, blue_val);
    %% MAKE IT A SINGLE COLOR
    color_mat = cat(3, ones(size(mip_center_2)),ones(size(mip_center_2)), zeros(size(mip_center_2)));
    
    hold on;
    h = imshow(color_mat);
    hold off;
    set(h, 'AlphaData', mip_center_2)
end


%% Add overlay of previous track
if opt == 8
    disp('hide');
else
    mip_center_1 = max(crop_blank_truth_2_PREV, [], 3);
    % CONNECT TO FORM LINE
    mip_center_1 = imdilate(mip_center_1, strel('disk', 10));
    mip_center_1 = bwskel(imbinarize(mip_center_1));
    mip_center_1 = imdilate(mip_center_1, strel('disk', 1));
    %figure; imshow(mip_center_1);
    
    blue = cat(3, zeros(size(mip_1)), zeros(size(mip_1)), ones(size(mip_1)));
    hold on;
    h = imshow(blue);
    hold off;
    set(h, 'AlphaData', mip_center_1)
end

%% add overlay of where cell is on the ORIGINAL timeframe
if opt == 8
    disp('hide');
else
    mip_center_1 = max(crop_blank_truth_1, [], 3);
    magenta = cat(3, zeros(size(mip_1)), ones(size(mip_1)), zeros(size(mip_1)));
    hold on;
    h = imshow(magenta);
    hold off;
    set(h, 'AlphaData', mip_center_1)
end



%% add overlay of current cell
if opt == 8
    disp('hide');
else
    mip_center_2 = max(crop_blank_truth_2, [], 3);
    magenta = cat(3, ones(size(mip_2)), zeros(size(mip_2)), ones(size(mip_2)));
    hold on;
    h = imshow(magenta);
    hold off;
    set(h, 'AlphaData', mip_center_2)
end

title(strcat('Correcting cell: ', {' '},num2str(check_neighbor), ' of total: ', {' '},num2str(length(neighbor_idx))))
text(-88, 0, strcat('Frame: ', {' '}, num2str(timeframe_idx + 1), ' of total: ', {' '},num2str(1)))

end