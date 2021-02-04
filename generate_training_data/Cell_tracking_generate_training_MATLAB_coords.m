
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



for fileNum = 1:length(natfnames_csv)
    cd(foldername);
    filename_raw = natfnames{1};
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
    
    
    
    %% read csv for comparison
    
    addpath('../cell_crop_func')

    %% Load csv as well
    filename_raw_csv = natfnames_csv{fileNum};
    syGlass10x = readtable(filename_raw_csv);
    
    %% START
    save_name = filename_raw;
    save_name = split(save_name, '0.tif');
    save_name = save_name{1};
    
    frame = syGlass10x.FRAME;
    series = syGlass10x.SERIES;
    all_Z = syGlass10x.Z;
    all_X = syGlass10x.Y;
    all_Y = syGlass10x.X;
    
    
    %% Cells to check
    % 506 / 460
    % 574 / 724 / 794
    % 336 / 349
    % 293 / 1514
    % 276 / 448
    %cells_to_check = [506, 460, 574, 724, 794, 336, 349, 293, 1514, 276, 448];
    %idx_cells = ismember(series, cells_to_check);
    %idx_cells = find(idx_cells);
    
%     frame = frame(idx_cells);
%     all_Z = all_Z(idx_cells);
%     all_X = all_X(idx_cells);
%     all_Y = all_Y(idx_cells);
%     series = series(idx_cells);
    
    
    %% Continue
    together = [frame, all_X, all_Y, all_Z, series];
    [~,idx] = sort(together(:,1)); % sort just the first column
    sortedmat = together(idx,:);   % sort the whole matrix using the sort indices
    
    cur_idx = 0;
    im_size = [im_x_size, im_y_size, im_z_size];
    blank_im = zeros(im_size);
    for i = 1:length(sortedmat)
        
        if cur_idx == sortedmat(i)
            % plot
            x = round(sortedmat(i, 2));
            y = round(sortedmat(i, 3));
            z = round(sortedmat(i, 4));
            
            lin_ind = sub2ind(size(blank_im), x, y, z);
            blank_im(lin_ind) = 1;
            %blank_im(lin_ind) = mod(sortedmat(i, 5), 5) + 100;
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