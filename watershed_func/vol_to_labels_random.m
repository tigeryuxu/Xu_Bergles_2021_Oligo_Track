function [labelled] = vol_to_labels_random(bw)
    cc = bwconncomp(bw);
    labelled = zeros(size(bw));
    for i = 1:length(cc(1).PixelIdxList)
        cur_idx = cc(1).PixelIdxList;

        %% Assigns random value
        num = randi([1 255]);
        labelled(cur_idx{i}) = num;
    end
end