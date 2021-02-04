function [option_num, matrix_timeseries, term] = Bergles_manual_correct(frame_1, frame_2, truth_1, truth_2, crop_frame_2, D, check_neighbor, neighbor_idx, matrix_timeseries, cur_timeseries, next_timeseries, timeframe_idx, x_min, x_max, y_min, y_max, z_min, z_max, crop_size, z_size,cur_centroids, next_centroids, dist_thresh, ssim_val_thresh, cur_cell_idx, total_cells_to_correct, total_num_frames, cur_centroids_scaled, next_centroids_scaled, dist_label_idx_matrix, x_first, y_first, z_first)

% Allows user to manually correct the counted the fully counted image:

term = 0;
% Plot
[ssim_val, dist] = plot_im(0);


%% 2nd check of ssim and distance metric with expanded crop
% if ssim_val > ssim_val_thresh && dist < dist_thresh
%     option_num = 10;
% else
%     option_num = 1;
% end
option_num = 1;

sorted_idx = 0;
option_new_cell = 0;
%% Now add selection menu


while option_num>0 && term == 0
    
    try
        if option_num == 10
            disp('skip and add')
            option_num = 1;
        else
            k = getkey(1,'non-ascii');
            option_num=str2double(k);
            if isnan(option_num)
                option_num = k;
            end
        end
        %% If key == 1, then is correctly matched cell
        if option_num==1
            if option_new_cell == 0
                next_cell = next_timeseries(neighbor_idx(check_neighbor));
                voxelIdxList = next_cell.objDAPI;
                centroid = next_cell.centerDAPI;
                cell_num = check_neighbor;
                % create cell object
                confidence_color = 1;
                cell_obj = cell_class(voxelIdxList,centroid, cell_num, confidence_color);
                matrix_timeseries{check_neighbor, timeframe_idx + 1} = cell_obj;
                
                %% add newly selected cell from option "a" below
            elseif option_new_cell == 1
                %disp('yeet')
                next_cell = next_timeseries(sorted_idx);
                voxelIdxList = next_cell.objDAPI;
                centroid = next_cell.centerDAPI;
                cell_num = check_neighbor;
                % create cell object
                confidence_color = 1;
                cell_obj = cell_class(voxelIdxList,centroid, cell_num, confidence_color);
                matrix_timeseries{check_neighbor, timeframe_idx + 1} =  cell_obj;
                
                sorted_idx = 0;
                option_new_cell = 0;
               
                %% add newly selected cell from option "3" below
            elseif option_new_cell == 2
                disp('yeet')
                option_new_cell = 0;
                %continue;  % because already added below
            end
            
            
            term = 1;
            break;
            
            %% If key == 2, then NOT the same cell, so don't include
        elseif option_num==2
            matrix_timeseries{check_neighbor, timeframe_idx + 1} = [];
            term = 2;
            
            %% If key == a, "add" then need to assign to NEW point
        elseif option_num=='a'
            
            cell_point=impoint;
            % Get XY position from placed dot
            poss_sub =getPosition(cell_point);
            
            % Get z-position from prompt
            prompt = {'Enter z-axis position:'};
            dlgtitle = 'slice position';
            definput = {'11'};
            answer = inputdlg(prompt,dlgtitle, [1, 35], definput);
            
            coordinates = [round(poss_sub), str2num(answer{1})];
            
            % create blank volume with only selected point
            blank_vol = zeros(size(crop_frame_2));
            blank_vol(sub2ind(size(crop_frame_2),coordinates(2), coordinates(1), coordinates(3))) = 1;
            
            % dilate to make matching easier
            blank_vol = imdilate(blank_vol, strel('sphere', 2));
            
            
            % insert point into larger volume
            full_vol = zeros(size(truth_2));
            full_vol(x_min:x_max, y_min:y_max, z_min:z_max) = blank_vol;
            
            % find index of point in relation to larger volume
            linear_idx = find(full_vol);
            
            % search through cells in next time frame to find which ones match
            matched = 0;
            for sorted_idx = 1:length(next_timeseries)
                if isempty(next_timeseries(sorted_idx).objDAPI)
                    continue;
                end
                sorted_cell = next_timeseries(sorted_idx).objDAPI;
                same = ismember(linear_idx, sorted_cell);
                
                % if matched
                if ~isempty(find(same, 1))
                    matched = 1;
                    break;
                end
            end
            
            % save corrected cell after sorting to matrix_timeseries
            if matched == 1
                option_new_cell = 1;
                disp('first yeet')
                % also, change the value in the next_centroids array (to plot updated)
                [x, y, z]= ind2sub(size(full_vol), linear_idx);
                %next_centroids(neighbor_idx(check_neighbor), :) = [y(round(length(y)/2)), x(round(length(x)/2)), z(round(length(z)/2))];
                
                % if did NOT match a point, reselect
            elseif matched == 0
                f = msgbox('Points did not match existing cell, please reselect');
            end
            
            delete(cell_point);
            plot_im(0);
            
            %% If key == d, "delete" then delete current cell on current timeframe (i.e. not a real cell)
        elseif option_num == 'd'
            matrix_timeseries{check_neighbor, timeframe_idx} = [];
            matrix_timeseries{check_neighbor, timeframe_idx + 1} = [];
            cur_centroids(check_neighbor, :) = [];
            
            term = 4;
            break;
            
            %% If key == s, "scale" then need to scale image outwards in the plot
        elseif option_num == 's'
            % Get scaling from prompt
            prompt = {'Enter desired XYZ scaling (0 - inf):'};
            dlgtitle = 'scaling requested';
            definput = {'2'};
            answer = inputdlg(prompt,dlgtitle, [1, 35], definput);
            scale_XYZ = str2num(answer{1});
            
            plot_im(scale_XYZ);
            
            %% If key == c, "clahe" then do CLAHE
        elseif option_num =='c'
            plot_im('adjust');
            
            
            %% If key == 3, add non-exisiting cell   ==> NOT YET WORKING
        elseif option_num ==3
            %plot_im(0);
            cell_point=drawpoint;
            % Get XY position from placed dot
            %poss_sub =getPosition(cell_point);
            poss_sub = cell_point.Position;
            
            % Get z-position from prompt
            prompt = {'Enter z-axis position:'};
            dlgtitle = 'slice position'
            definput = {'10'};
            answer = inputdlg(prompt,dlgtitle, [1, 35], definput);
            
            coordinates = [round(poss_sub), str2num(answer{1})];
            
            % create blank volume with only selected point
            blank_vol = zeros(size(crop_frame_2));
            blank_vol(sub2ind(size(crop_frame_2),coordinates(2), coordinates(1), coordinates(3))) = 1;
            
            % dilate to make matching easier
            blank_vol = imdilate(blank_vol, strel('sphere', 2));
            
            % insert point into larger volume
            full_vol = zeros(size(truth_2));
            full_vol(x_min:x_max, y_min:y_max, z_min:z_max) = blank_vol;
            
            % find index of point in relation to larger volume
            linear_idx = find(full_vol);
            
            
            %% Add dilated cell point into the matrix_timeseries!
            %next_cell = next_timeseries(neighbor_idx(check_neighbor));
            voxelIdxList = linear_idx;
            [x, y, z]= ind2sub(size(full_vol), linear_idx);
            centroid = [y(round(length(y)/2)), x(round(length(x)/2)), z(round(length(z)/2))];
            cell_num = check_neighbor;
            % create cell object
            confidence_color = 1;
            cell_obj = cell_class(voxelIdxList,centroid, cell_num, confidence_color);
            matrix_timeseries{check_neighbor, timeframe_idx + 1} = cell_obj;
            
            
            %% set point as this new one in the points array for plotting
            %next_centroids(neighbor_idx(check_neighbor), :) = [y(round(length(y)/2)), x(round(length(x)/2)), z(round(length(z)/2))];
            
            option_new_cell = 2;
            
            %% Plot how it looks now
            delete(cell_point);
            plot_im(0);
                        
            %% If key == h, then hide overlay for ease of view
        elseif option_num=='h'
            plot_im(8);
            
            
            
                  
            %% If key == n, "delete" then delete current cell on current timeframe (i.e. not a real cell)
        elseif option_num == 'n'
            matrix_timeseries{check_neighbor, timeframe_idx} = [];
            %matrix_timeseries{check_neighbor, timeframe_idx + 1} = [];
            cur_centroids(check_neighbor, :) = [];
            
            %plot_im(0);
            cell_point=drawpoint;
            % Get XY position from placed dot
            %poss_sub =getPosition(cell_point);
            poss_sub = cell_point.Position;
            
            % Get z-position from prompt
            prompt = {'Enter z-axis position:'};
            dlgtitle = 'slice position'
            definput = {'10'};
            answer = inputdlg(prompt,dlgtitle, [1, 35], definput);
            
            coordinates = [round(poss_sub), str2num(answer{1})];
            
            % create blank volume with only selected point
            blank_vol = zeros(size(crop_frame_2));
            blank_vol(sub2ind(size(crop_frame_2),coordinates(2), coordinates(1), coordinates(3))) = 1;
            
            % dilate to make matching easier
            blank_vol = imdilate(blank_vol, strel('sphere', 2));
            
            % insert point into larger volume
            full_vol = zeros(size(truth_1));
            full_vol(x_min:x_max, y_min:y_max, z_min:z_max) = blank_vol;
            
            % find index of point in relation to larger volume
            linear_idx = find(full_vol);
            
            
            %% Add dilated cell point into the matrix_timeseries!
            %next_cell = next_timeseries(neighbor_idx(check_neighbor));
            voxelIdxList = linear_idx;
            [x, y, z]= ind2sub(size(full_vol), linear_idx);
            centroid = [y(round(length(y)/2)), x(round(length(x)/2)), z(round(length(z)/2))];
            cell_num = check_neighbor;
            % create cell object
            confidence_color = 1;
            cell_obj = cell_class(voxelIdxList,centroid, cell_num, confidence_color);
            matrix_timeseries{check_neighbor, timeframe_idx} = cell_obj;
            
            
            %% set point as this new one in the points array for plotting
            %next_centroids(neighbor_idx(check_neighbor), :) = [y(round(length(y)/2)), x(round(length(x)/2)), z(round(length(z)/2))];
            
            option_new_cell = 2;
            
            %% Plot how it looks now
            delete(cell_point);
            plot_im(0);

        elseif option_num == 'backspace'
                        
            term = 10;
            break;
        elseif option_num == 'escape'
            term = 99;
            break;
            
        else
            waitfor(msgbox('Key did not match any command, please reselect'));
            plot_im(0);
            option_num = 100;
            %pause;
            
        end
    catch
        continue;
    end
    
end


%% TO PLOT the wholeimage
    function [ssim_val, dist] = plot_im(opt)
        
        scale_XYZ = opt;
        if opt == 'adjust'
            disp('adjust');
        elseif opt == 8
            disp('hide');
        elseif opt == 10
            disp('skip pause')
        elseif opt > 0
            scale_XYZ = opt;
            original_size = crop_size;
            original_z = z_size;
            crop_size = crop_size * scale_XYZ;
            z_size = z_size * scale_XYZ;
        end
                
        %% Switched to plotting with crops relative to ORIGINAL timeframe
        crop_size = round(crop_size);
        z_size = round(z_size);
        
        %% Parse frame 1
        if ~isempty(matrix_timeseries{check_neighbor, timeframe_idx })
            frame_1_centroid = matrix_timeseries{check_neighbor, timeframe_idx}.centroid;
        else
            frame_1_centroid = cur_centroids(check_neighbor, :);
        end
        % get frame 1 crop
        y = round(frame_1_centroid(1)); x = round(frame_1_centroid(2)); z = round(frame_1_centroid(3));
        im_size = size(frame_1);
        height = im_size(1);  width = im_size(2); depth = im_size(3);
        crop_frame_1 = crop_around_centroid(frame_1, y, x, z, crop_size, z_size, height, width, depth);
        crop_truth_1 = crop_around_centroid(truth_1, y, x, z, crop_size, z_size, height, width, depth);
        
        % get a truth with ONLY the current cell of interest
        blank_truth = zeros(size(truth_1));
        blank_truth(x, y, z) = 1;
        crop_blank_truth_1 = crop_around_centroid(blank_truth, y, x, z, crop_size, z_size, height, width, depth);
        crop_blank_truth_1 = imdilate(crop_blank_truth_1, strel('sphere', 2));
        
        mip_1 = max(crop_frame_1, [], 3);
        %subplot(1, 2, 1); imshow(mip_1);
        
        %% Parse frame 2 and plot ONLY if it exists (b/c may have been deleted on return
        %frame_2_centroid = next_centroids(neighbor_idx(check_neighbor), :);
       
        if ~isempty(matrix_timeseries{check_neighbor, timeframe_idx + 1})
            frame_2_centroid = matrix_timeseries{check_neighbor, timeframe_idx + 1}.centroid;
            
            y_2 = round(frame_2_centroid(1)); x_2 = round(frame_2_centroid(2)); z_2 = round(frame_2_centroid(3));
            blank_truth = zeros(size(truth_2));
            blank_truth(x_2, y_2, z_2) = 1;
            
        else
            blank_truth = zeros(size(truth_2));
            blank_truth(x_first, y_first, z_first) = 1;
        end
        
        %% Do not center
        im_size = size(frame_2);
        height = im_size(1);  width = im_size(2); depth = im_size(3);
        [crop_frame_2, x_min, x_max, y_min, y_max, z_min, z_max] = crop_around_centroid(frame_2, y, x, z, crop_size, z_size, height, width, depth);
        crop_truth_2 = crop_around_centroid(truth_2, y, x, z, crop_size, z_size, height, width, depth);
                
        %% get a truth with ONLY the current cell of interest

        %blank_truth = imdilate(blank_truth, strel('sphere', 3));
        crop_blank_truth_2 = crop_around_centroid(blank_truth, y, x, z, crop_size, z_size, height, width, depth);
        crop_blank_truth_2 = imdilate(crop_blank_truth_2, strel('sphere', 2));

        mip_2 = max(crop_frame_2, [], 3);
        %subplot(1, 2, 2); imshow(mip_2); 
        
        
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
        
        %% Connect the dots and make XY and XZ projections
        crop_blank_XY = max(crop_blank_truth_2_PREV, [], 3);
        cc = bwconncomp(crop_blank_XY);
        centers = regionprops(cc, 'Centroid');
        for center_idx = 1:length(centers) - 1
           cur_centers = round(centers(center_idx).Centroid);
           next_centers = round(centers(center_idx + 1).Centroid);
           
           crop_blank_XY = func_DrawLine(crop_blank_XY, cur_centers(2), cur_centers(1), next_centers(2), next_centers(1), 1);
        end
        crop_blank_XY = imdilate(crop_blank_XY, strel('disk', 1));
        
        crop_blank_XZ = squeeze(max(crop_blank_truth_2_PREV, [], 1));
        crop_blank_XZ = permute(crop_blank_XZ, [2 1]);
        cc = bwconncomp(crop_blank_XZ);
        centers = regionprops(cc, 'Centroid');
        for center_idx = 1:length(centers) - 1
             cur_centers = round(centers(center_idx).Centroid);
             next_centers = round(centers(center_idx + 1).Centroid);
            
            crop_blank_XZ = func_DrawLine(crop_blank_XZ, cur_centers(2), cur_centers(1), next_centers(2), next_centers(1), 1);
        end
        crop_blank_XZ = imdilate(crop_blank_XZ, strel('disk', 1));
                
        
        %% accuracy metrics
        dist = D(check_neighbor);
        ssim_val = ssim(crop_frame_1, crop_frame_2);

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
            if isempty( matrix_timeseries{cell_idx, timeframe_idx})
                continue
            end
            if isempty(matrix_timeseries{cell_idx, timeframe_idx + 1})
                continue;
            end
            % add the 2nd frame color at the same time
            cur_cell = matrix_timeseries{cell_idx, timeframe_idx + 1};
            voxels = cur_cell.voxelIdxList;
            im_frame_2(voxels) = cell_idx;
           
            
            cur_cell = matrix_timeseries{cell_idx, timeframe_idx};
            voxels = cur_cell.voxelIdxList;
            im_frame(voxels) = cell_idx;
        end
        
        %% MAKE IT A SINGLE COLOR
        labels_crop_truth_1 = crop_around_centroid(im_frame, y, x, z, crop_size, z_size, height, width, depth);
        mip_center_1 = max(labels_crop_truth_1, [], 3);
        
        color_mat = cat(3, ones(size(mip_center_1)),ones(size(mip_center_1)), zeros(size(mip_center_1)));
        
        hold on;
        h = imshow(color_mat);
        hold off;
        set(h, 'AlphaData', mip_center_1)
      
 
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
        text(-88, 0, strcat('Frame: ',{' '}, num2str(timeframe_idx), ' of total: ', {' '},num2str(total_num_frames)))
        
        text(-88, 20, strcat('Green dot:'))
        text(-88, 30, strcat('location on left frame') )
        
        text(-88, 50, strcat('Magenta dot:'))
        text(-88, 60, strcat('location on right frame') )
        
        text(-88, 80, strcat('Blue line:'))
        text(-88, 90, strcat('loc on previous frames') )
        
        text(-88, 110, strcat('Current top slice:', {' '}, num2str(z - z_size/2)))
        
        %% (2) Plot XZ view of right frame + track of line movement
        %% ***MAKE SURE THE SCALING IS ALWAYS ACCURATE
        
        middle = uipanel('Parent',p, 'Position', [.367 0.2 .28 .20]);
        % set 0 to things +/- 20 pixels away from the current centroid to
        % clear up the XZ projection a bit
        crop_frame_2_temp = crop_frame_2;
        location = find(crop_blank_truth_1);
        [a, b, c] = ind2sub(size(crop_blank_truth_1), location);
        x_center = round(mean(a));
        pad = 30; 
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
            mip_center_1 = crop_blank_XZ;
            XZ_size = size(mip_center_1);
            mip_center_1 = imresize(mip_center_1, [XZ_size(1) * 3, XZ_size(2) * 0.83]);
           
            % CONNECT TO FORM LINE
            %mip_center_1 = imdilate(mip_center_1, strel('disk', 2));
            %mip_center_1 = bwskel(imbinarize(mip_center_1));
            %mip_center_1 = imdilate(mip_center_1, strel('disk', 1));
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
        
        
        title('RIGHT frame: Scaled XZ project + tracking (centered crop +/- 30 px)')
        
        
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
            mip_center_1 = crop_blank_XZ;
            XZ_size = size(mip_center_1);
            mip_center_1 = imresize(mip_center_1, [XZ_size(1) * 3, XZ_size(2) * 0.83]);
            
            % CONNECT TO FORM LINE
            %mip_center_1 = imdilate(mip_center_1, strel('disk', 2));
            %mip_center_1 = bwskel(imbinarize(mip_center_1));
            %mip_center_1 = imdilate(mip_center_1, strel('disk', 1));
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
        

        %% (3) Add graph of vectors
        top_right_corner = uipanel('Parent',p, 'Position', [.85 0.65 .15 .3]);
        ax = axes('parent', top_right_corner);
        
        plot_bool = 1;
        skip = 1;
        % skip this if less than 5 cells to get vectors from
        hold on;
        [avg_vec, all_unit_v, all_dist_to_avg, cells_in_crop] = find_avg_vectors(dist_label_idx_matrix...
            , cur_timeseries ,frame_1, crop_size, z_size, cur_centroids...
            ,cur_centroids_scaled, next_centroids_scaled...
            , check_neighbor, neighbor_idx, plot_bool, skip);
        
        % (3) get current vector
        outlier_vec_bool = []; outlier_vec_bool_95 = [];
        if skip && length(cells_in_crop) > 5
            cell_of_interest =  cur_centroids_scaled(check_neighbor, :);
            neighbor_of_cell =  next_centroids_scaled(neighbor_idx(check_neighbor), :);
            vector = abs(cell_of_interest) - abs(neighbor_of_cell);
            %unit_v_check = vector/norm(vector);
            if plot_bool
                plot3([0, vector(1)], [0, vector(2)], [0, vector(3)], 'LineWidth', 10);
            end
            dist_to_avg = avg_vec - vector;
            dist_to_avg = norm(dist_to_avg);
            
            % (4) check if it is an outlier to the 90th percentile:
            outliers = find(isoutlier(all_dist_to_avg, 'percentiles', [0, 90]));
            outliers_idx = cells_in_crop(outliers);
            
            outlier_vec_bool = find(ismember(outliers_idx, check_neighbor));
            
            % (5) check if it is a LARGE outlier to the 95th percentile:
            outliers = find(isoutlier(all_dist_to_avg, 'percentiles', [0, 95]));
            outliers_idx = cells_in_crop(outliers);
            
            outlier_vec_bool_95 = find(ismember(outliers_idx, check_neighbor));
            
            
            %% Tiger added new: because in cuprizone many neighbors are badly associated, so need to set some extra checks
            if dist_to_avg > 20
                outlier_vec_bool = check_neighbor;
            end
            
            
        end
        

        %% CLEAR the plot if NOT an outlier!!!
        if isempty(outlier_vec_bool)
            top_right_corner = uipanel('Parent',p, 'Position', [.85 0.65 .15 .3]);
            ax = axes('parent', top_right_corner);
            axis off
            title('Insufficient cells or movement')
        elseif ~isempty(outlier_vec_bool_95)
             title('movmt vectors 95th percentile');
        else
              title('movmt vectors 90th percentile');  
        end

    
        %% (4) Plot bottom RIGHT graph - plot max
        if opt == 'adjust'
            mip_2 = adapthisteq(mip_2);
        end
        ax = axes('parent', bottom_right);
        imshow(mip_2);
        colormap('gray'); axis off
        %% Add overlay of tracked cells with diff colors
        labels_crop_truth_2 = crop_around_centroid(im_frame_2, y, x, z, crop_size, z_size, height, width, depth);
        
        %% MAKE IT A SINGLE COLOR
        mip_center_2 = max(labels_crop_truth_2, [], 3);
        color_mat = cat(3, ones(size(mip_center_2)),ones(size(mip_center_2)), zeros(size(mip_center_2)));
        
        hold on;
        h = imshow(color_mat);
        hold off;
        set(h, 'AlphaData', mip_center_2)
        
        
        %% Add overlay of previous track
        if opt == 8
            disp('hide');
        else
            mip_center_1 = crop_blank_XY;
            % CONNECT TO FORM LINE
            %mip_center_1 = imdilate(mip_center_1, strel('disk', 10));
            %mip_center_1 = bwskel(imbinarize(mip_center_1));
            %mip_center_1 = imdilate(mip_center_1, strel('disk', 1));
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
        
        title(strcat('Correcting cell: ', {' '},num2str(cur_cell_idx), ' of total: ', {' '},num2str(total_cells_to_correct)))
        text(-88, 0, strcat('Frame: ', {' '}, num2str(timeframe_idx + 1), ' of total: ', {' '},num2str(total_num_frames)))
        

  
        % restore original crop size
        if opt == 'adjust'
            disp('adjust');
        elseif opt == 8
            disp('hide');
        elseif opt == 10
            disp('skip pause')
        elseif opt > 0
            crop_size = original_size;
            z_size = original_z;
        end
        
        
        % pause allows usage of the scroll bar
        pause
   
        
        
    end


end