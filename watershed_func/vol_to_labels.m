function [labelled] = vol_to_labels(bw)
    cc = bwconncomp(bw);
    labelled = zeros(size(bw));
    for i = 1:length(cc(1).PixelIdxList)
        cur_idx = cc(1).PixelIdxList;
       labelled(cur_idx{i}) = i; 
    end
end