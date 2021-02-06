function [crop, box_x_min ,box_x_max, box_y_min, box_y_max, box_z_min, box_z_max] = crop_around_centroid(input_im, y, x, z, crop_size, z_size, height, width, depth)
box_x_max = x + crop_size; box_x_min = x - crop_size;
box_y_max = y + crop_size; box_y_min = y - crop_size;
box_z_max = z + z_size/2; box_z_min = z - z_size/2;

im_size_x = height;
im_size_y = width;
im_size_z = depth;

if box_x_max > im_size_x
    overshoot = box_x_max - im_size_x;
    box_x_max = box_x_max - overshoot;
    box_x_min = box_x_min - overshoot;
end

if box_x_min <= 0
    overshoot_neg = (-1) * box_x_min + 1;
    box_x_min = box_x_min + overshoot_neg;
    box_x_max = box_x_max + overshoot_neg;
end


if box_y_max > im_size_y
    overshoot = box_y_max - im_size_y;
    box_y_max = box_y_max - overshoot;
    box_y_min = box_y_min - overshoot;
end

if box_y_min <= 0
    overshoot_neg = (-1) * box_y_min + 1;
    box_y_min = box_y_min + overshoot_neg;
    box_y_max = box_y_max + overshoot_neg;
end



if box_z_max > im_size_z
    overshoot = box_z_max - im_size_z;
    box_z_max = box_z_max - overshoot;
    box_z_min = box_z_min - overshoot;
end

if box_z_min <= 0
    overshoot_neg = (-1) * box_z_min + 1;
    box_z_min = box_z_min + overshoot_neg;
    box_z_max = box_z_max + overshoot_neg;
end

box_x_max - box_x_min;
box_y_max - box_y_min;
box_z_max - box_z_min;


crop = input_im(box_x_min + 1:box_x_max, box_y_min + 1:box_y_max, box_z_min + 1:box_z_max);
end


