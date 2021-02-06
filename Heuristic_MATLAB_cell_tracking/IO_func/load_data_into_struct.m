function [all_s, frame, truth, og_size] = load_data_into_struct(foldername, natfnames, fileNum, all_s, thresh_size, first_slice, last_slice)

%% Load input data
filename_raw = natfnames{fileNum};
cd(foldername);
[frame] = load_3D_gray(filename_raw);
og_size = size(frame);

frame = frame(:, :, first_slice:last_slice);

%% Also load truth
filename_raw = natfnames{fileNum + 1};
cd(foldername);
[truth] = load_3D_gray(filename_raw);
truth(truth > 0) = 1;

truth = truth(:, :, first_slice:last_slice);


%% Eliminate synapses on edges
%     new_im = zeros(size(truth_binary));
%     cc = bwconncomp(truth_binary);
%     objs = regionprops3(cc, 'VoxelList', 'VoxelIdxList');
%     objs_voxels = objs.VoxelList;
%     objs_idxList = objs.VoxelIdxList;
%     for obj_idx = 1:length(objs_voxels)
%         cur_obj = objs_voxels{obj_idx};
%         max_z_slice = max(cur_obj(:, 3));
%         min_z_slice = min(cur_obj(:, 3));
%
%         if (max_z_slice == 1 && min_z_slice == 1) || (max_z_slice == length(truth_binary(1, 1, :)) &&  min_z_slice == length(truth_binary(1, 1, :)))
%             disp('elim\n')
%             continue
%         end
%
%         new_im(objs_idxList{obj_idx}) = 1;
%     end
%     truth_binary = new_im;


%% watershed
%DAPIsize = 0; DAPImetric = 0;
%enhance_DAPI = 'N'; DAPI_bb_size = 50; binary = 'Y';
%[mat, objDAPI, DAPI_bw, labels, volume] = DAPIcount_3D(truth, DAPIsize, DAPImetric, enhance_DAPI, DAPI_bb_size, binary);  % function

%% without watershed
cc = bwconncomp(truth);
stats = regionprops3(cc,'Volume','Centroid', 'VoxelIdxList'); %%%***good way to get info about region!!!
volume = stats.Volume;
objDAPI = cell(1); idx = 1; mat = [];
for k = 1:length(stats.VoxelIdxList)
    centroid = stats.Centroid(k, :);
    mat = [mat ; centroid];       % add to matrix of indexes
    objDAPI{idx} = stats.VoxelIdxList{k};
    idx = idx + 1;
end

%% Initializes struct to store everything
c= cell(length(objDAPI), 1); % initializes Bool_W with all zeros
[c{:}] = deal(0);
strucMat = num2cell(mat, 2);
s = struct('objDAPI', objDAPI', 'centerDAPI', strucMat, 'Core', cell(length(objDAPI), 1)...
    , 'centroids', c, 'im_num', c, 'volume', num2cell(volume), 'im_size', c, 'OtherStats', c);
s(1).im_size = size(frame);

all_s{end + 1} = s;

%% Delete everything that is too small
for struct_idx = 1:length(all_s)
    cur_s = all_s{struct_idx};
    for cur_cell_idx = length(cur_s):-1:1
        cur_volume = cur_s(cur_cell_idx).volume;
        if cur_volume < thresh_size
            all_s{struct_idx}(cur_cell_idx) = [];
        end
    end
end

%% replot full figure with size thresholded image
% need this generic function for end as well!
truth = zeros(size(truth));
s = all_s{end};
for struct_idx = 1:length(s)
   cur_voxels = s(struct_idx).objDAPI; 
   truth(cur_voxels) = 1; 
   
end



end
