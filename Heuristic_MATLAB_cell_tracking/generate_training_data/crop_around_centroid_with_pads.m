function crop = crop_around_centroid_with_pads(input_im, y, x, z, crop_size, z_size, height, width, depth)

%% Tiger added rounding - June 19th, 2020 - but should maybe only round up???
box_x_max = round(x + crop_size); box_x_min = round(x - crop_size);
box_y_max = round(y + crop_size); box_y_min = round(y - crop_size);
box_z_max = round(z + z_size/2); box_z_min = round(z - z_size/2);

im_size_x = height;
im_size_y = width;
im_size_z = depth;

overshoot_x = 0; overshoot_neg_x = 0;
if box_x_max > im_size_x
    overshoot_x = box_x_max - im_size_x;
    box_x_max = im_size_x;
end

if box_x_min <= 0
    overshoot_neg_x = (-1) * (box_x_min - 1);  % b/c if it's zero, then already overshot by 1
    box_x_min = 1;
end

overshoot_y = 0; overshoot_neg_y = 0;
if box_y_max > im_size_y
    overshoot_y = box_y_max - im_size_y;
    box_y_max = im_size_y;
end

if box_y_min <= 0
    overshoot_neg_y = (-1) * (box_y_min - 1);  % b/c if it's zero, then already overshot by 1
    box_y_min = 1;
end


overshoot_z = 0; overshoot_neg_z = 0;
if box_z_max > im_size_z
    overshoot_z = box_z_max - im_size_z;
    box_z_max = im_size_z;
end


if box_z_min <= 0
    overshoot_neg_z = (-1) * (box_z_min - 1);  % b/c if it's zero, then already overshot by 1
    box_z_min = 1;
end

% overshoot_neg_x =  round(crop_size * 2) - (box_x_max - box_x_min);
% overshoot_neg_y =  round(crop_size * 2) -  (box_y_max - box_y_min);
% overshoot_neg_z =  round(z_size) - (box_z_max - box_z_min);


crop = input_im(box_x_min + 1:box_x_max, box_y_min + 1:box_y_max, box_z_min + 1:box_z_max);


sizes = size(crop);
crop_xs = sizes(1); crop_ys = sizes(2); crop_zs = sizes(3);

to_pad = zeros([round(crop_size) * 2, round(crop_size) * 2, round(z_size)]);

if overshoot_neg_x > 0
    x_pad_min = overshoot_neg_x + 1; %%% still need to +1 b/c index starts at 1
    x_pad_high = crop_xs + overshoot_neg_x;
else
    x_pad_min = 1;
    x_pad_high = crop_xs;
end

if overshoot_neg_y > 0
    y_pad_min = overshoot_neg_y + 1;
    y_pad_high = crop_ys + overshoot_neg_y;
else
    y_pad_min = 1;
    y_pad_high = crop_ys;
end


if overshoot_neg_z > 0
    z_pad_min = overshoot_neg_z + 1;
    z_pad_high = crop_zs + overshoot_neg_z;
else
    z_pad_min = 1;
    z_pad_high = crop_zs;
end

to_pad(x_pad_min:x_pad_high , y_pad_min:y_pad_high, z_pad_min:z_pad_high) = crop;

crop = to_pad;

end


