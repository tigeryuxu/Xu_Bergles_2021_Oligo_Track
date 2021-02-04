
opengl hardware;
close all;

addpath('../IO_func/')
addpath('../cell_crop_func/')

cur_dir = pwd;
addpath(strcat(cur_dir))  % adds path to functions
cd(cur_dir);

%% Initialize
foldername = uigetdir();   % get directory

%% Run Analysis
cd(foldername);   % switch directories
nameCat = '*.tif';
fnames = dir(nameCat);

trialNames = {fnames.name};
numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently

cd(cur_dir);
natfnames=natsort(trialNames);



%% Also get "csv"s
cd(foldername);   % switch directories
nameCat_csv = '*.csv*';
fnames_csv = dir(nameCat_csv);

trialNames_csv = {fnames_csv.name};
numfids_csv = length(trialNames_csv);   %%% divided by 5 b/c 5 files per pack currently

cd(cur_dir);
natfnames_csv =natsort(trialNames_csv);



for fileNum = 1:length(natfnames)

    cd(foldername);
    filename_raw = natfnames{fileNum};
    [red_3D, xyres] = load_3D_gray(filename_raw, natfnames);
    z_scale = 1/3;   % is in pixel/um axial depth
    x_scale = xyres;  % is in pixel/um
    y_scale = xyres;
        
    if isempty(xyres)
        x_scale = 1/0.8302662;
        y_scale = 1/0.8302662;
    end
    
    im_size = size(red_3D);
    im_y_size = im_size(2);
    im_x_size = im_size(1);
    im_z_size = im_size(3);
    
    
    %% Load csv as well
    filename_raw_csv = natfnames_csv{fileNum};
    syGlass10x = readtable(filename_raw_csv);
    
    % 680 == 1024, 1024, 185 ==> 10x ==> MOBPF_190105w_1_cuprBZA_10x_T=
    % 650 == 1052, 1036, 191 ==> 10x ==> MOBPF_190106w_5_cuprBZA_10x.tif - T=
    % also + 10 to X, - 5 to Y
    
    % 670 == 512, 512, 203 ==> 20x ==> MOBPF_190112w_8_cuprBZA_10x_cropped1quad.tif - T=
    
    %% START
    
    %im_y_size = 1024
    %im_x_size = 1024
    %im_z_size = 185
    save_name = filename_raw;
    save_name = split(save_name, '0.tif');
    save_name = save_name{1};
    
    % save_name = 'MOBPF_190106w_5_cuprBZA_10x.tif - T='
    %
    % im_y_size = 512
    % im_x_size = 512
    % im_z_size = 203
    % save_name = 'MOBPF_190112w_8_cuprBZA_10x_cropped1quad.tif - T='
    
    
    frame = syGlass10x.FRAME;
    all_Z = syGlass10x.Z;
    all_X = syGlass10x.Y;
    all_Y = syGlass10x.X;
    
    % frame = a1901128r670syGlass20x.FRAME;
    % all_Z = a1901128r670syGlass20x.Z;
    % all_Y = a1901128r670syGlass20x.Y;
    % all_X = a1901128r670syGlass20x.X;
    
    %x_um = 859.18 % in um for 10x resolution
    % x_um = max(all_X) + 25 - min(all_X);
    % x_scale = im_x_size ./ x_um;
    %
    % y_um = max(all_Y) + 25 - min(all_Y);
    % y_scale = im_y_size ./ y_um;
    
    %x_scale = 1.1918 %in um for 10x resolution 680
    %y_scale = 1.1918 %in um for 10x resolution
    
    %x_scale = 1.2044 %in um for 10x resolution 650
    %y_scale = 1.2044 %in um for 10x resolution
    
    %x_scale = 1.2044 %in um for 20x resolution 650
    %y_scale = 1.2044 %in um for 20x resolution
    
    
    %% Scale x and y
    all_X = all_X * x_scale;
    all_Y = all_Y * y_scale;
    
    %% Normalize to first val 0 indexing
    middle_val = im_x_size ./ 2;
    all_X = round(all_X + middle_val);
    %all_X = round(all_X + middle_val + 1);
    %all_X = round(all_X + middle_val + 10);
    
    middle_val = im_y_size ./ 2;
    all_Y = round(all_Y + middle_val);
    %all_Y = round(all_Y + middle_val - 5);
    
    

    
    %% Scale Z
    %z_um = max(all_Z) - min(all_Z);   % in um
    %z_um = 290 - min(all_Z);   % in um
    %z_scale = im_z_size ./ z_um;
        
    all_Z = all_Z * z_scale;
    middle_val = im_z_size ./ 2;
    all_Z = round(all_Z + middle_val);
    %all_Z = round(all_Z + middle_val + 5);
    
    together = [frame, all_X, all_Y, all_Z];
    [~,idx] = sort(together(:,1)); % sort just the first column
    sortedmat = together(idx,:);   % sort the whole matrix using the sort indices
    
    cur_idx = 0;
    im_size = [im_x_size, im_y_size, im_z_size];
    blank_im = zeros(im_size);
    for i = 1:length(sortedmat)
        
        if cur_idx == sortedmat(i)
            % plot
            x = sortedmat(i, 2);
            if x > im_x_size
                x = im_x_size
            elseif x <= 0
                x = 1;
            end

            y = sortedmat(i, 3);
            if y > im_y_size
                y = im_y_size
            elseif y <= 0
                y = 1;
            end
            z = sortedmat(i, 4);
            if z > im_z_size
                z = im_z_size
            elseif z <= 0
                z = 1;
            end
            
            lin_ind = sub2ind(size(blank_im), x, y, z);
            blank_im(lin_ind) = 1;
            
        else
            cur_idx = cur_idx + 1;
            %figure(); volshow(blank_im);
            blank_im = imdilate(blank_im, strel('sphere', 4));
            % save image as well
            for k = 1:length(blank_im(1, 1, :))
                im_2D = blank_im(:, :, k);
                im_2D = im2uint8(im_2D);
                
                %figure(888); imshow(im_2D);
                filename_save = save_name;
                imwrite(im_2D, strcat(filename_save, num2str(cur_idx - 1),'_truth.tif') , 'writemode', 'append', 'Compression','none')
            end
            
            % then create new blank
            blank_im = zeros(im_size);
            %break;
        end
        
        
    end
    
    
    %% Print out the last frame
    cur_idx = cur_idx + 1;
    blank_im = imdilate(blank_im, strel('sphere', 4));
    % save image as well
    for k = 1:length(blank_im(1, 1, :))
        im_2D = blank_im(:, :, k);
        im_2D = im2uint8(im_2D);
        
        %figure(888); imshow(im_2D);
        filename_save = save_name;
        imwrite(im_2D, strcat(filename_save, num2str(cur_idx - 1),'_truth.tif') , 'writemode', 'append', 'Compression','none')
    end
    
    % then create new blank
    blank_im = zeros(im_size);
    
    
    
end