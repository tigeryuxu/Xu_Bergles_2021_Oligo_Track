% get average vector of cells bw timepoints
function vectorPerTP = getVector(series)
    tp = fieldnames(series);
    vectorPerTP = NaN(length(tp)-1,3);
    for k = 2:length(tp)
        cells_next = series.(tp{k})(:,1);
        cells_prev = series.(tp{k-1})(:,1);
        vector = NaN(1000,3); % preallocated, NaNs removed in mean function
        for i = 1:length(cells_next)
            idx = cells_prev==cells_next(i);
            if sum(idx)
                coordsN = series.(tp{k})(i,end-2:end);
                coordsP = series.(tp{k-1})(idx,end-2:end);
                if size(coordsP,1) > 1
                    temp = series.(tp{k-1})(idx,1);
                    warning('%d is duplicated',temp(1));
                    continue
                end
                vector(i,:) = coordsN - coordsP; 
            end
        end
        vectorPerTP(k-1,:) = mean(vector,1,'omitnan');
    end
end