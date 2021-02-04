function [matrix_timeseries] = Bergles_manual_correct_last_frame(frame_num, frame_1, truth_1, matrix_timeseries, crop_size, z_size)

% Allows user to manually correct the counted the fully counted image:

for i = 1:length(matrix_timeseries(:, frame_num))

    if frame_num == 1 && ~isempty(matrix_timeseries{i, frame_num}) 
        disp('first frame')
    elseif isempty(matrix_timeseries{i, frame_num}) || ~isempty(matrix_timeseries{i, frame_num - 1})
        continue;
    end
    
    %% get crops
    cur_cell = matrix_timeseries{i, frame_num};
    frame_1_centroid = cur_cell.centroid;
    close all;
    im_size = size(frame_1);
    height = im_size(1);  width = im_size(2); depth = im_size(3);
    y = round(frame_1_centroid(1)); x = round(frame_1_centroid(2)); z = round(frame_1_centroid(3));
    [crop_frame_1, box_x_min ,box_x_max, box_y_min, box_y_max, box_z_min, box_z_max] = crop_around_centroid(frame_1, y, x, z, crop_size, z_size, height, width, depth);
    
    blank_truth = zeros(size(truth_1));
    blank_truth(x, y, z) = 1;
    %blank_truth = imdilate(blank_truth, strel('sphere', 3));
    crop_blank_truth_1 = crop_around_centroid(blank_truth, y, x, z, crop_size, z_size, height, width, depth);
    crop_blank_truth_1 = imdilate(crop_blank_truth_1, strel('sphere', 2));
    
    mip_1 = max(crop_frame_1, [], 3); 
    figure(5);
       % txt = {'Check newly added cells are real and also elim from edges.', 'Press: "1" to Keep, "2" to Elim'};
   % text(0,0, txt)
    subplot(1,2,1);
    imshow(mip_1);
    %colormap('gray'); axis off
    
    subplot(1, 2, 2);
    imshow(mip_1);
    mip_center_1 = max(crop_blank_truth_1, [], 3);
    magenta = cat(3, ones(size(mip_1)), zeros(size(mip_1)), ones(size(mip_1)));
    hold on;
    h = imshow(magenta);
    hold off;
    set(h, 'AlphaData', mip_center_1)
    
    
    txt = {'Double check newly added cells are real and also elim from edges of image', 'Press: "1" to Keep, "2" to Elim'};
    text(-250,250, txt)
    pause
    
    try
        k = getkey(1,'non-ascii');
        option_num=str2double(k);
        if isnan(option_num)
            option_num = k;
        end
        
        %% If key == 1, then is correctly matched cell
        if option_num==1
            continue;
  
            %% If key == 2, then NOT the same cell, so must delete from matrix_timeseries
        elseif option_num==2
            %matrix_timeseries{i, frame_num} = [];
       
            matrix_timeseries(i, :) = {[]};   % set entire row to nothing
        else
            waitfor(msgbox('Key did not match any command, please1 please reselect'));
            plot_im(0);
            option_num = 100;
            %pause;
            
        end
    catch
        continue;
    end
    
end

