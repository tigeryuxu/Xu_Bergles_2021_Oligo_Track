function [Dstats, vectorTot, bslnCoords, lastCoords, lastNewCoords] = calculateSomaKNN_ctrl(~, XLS)
% KNN analysis of baseline vs last tp and compare average distance to baseline vs rotated 90 deg last tp
%% initialize
maxtp = max(XLS(:,3));
% make structure called series with fields for each time point (e.g. tp1, tp2, etc)
for i = 0:maxtp
    str = ['tp' num2str(i)];
    series.(str) = XLS(XLS(:,3) == i, :);
end
tp = fieldnames(series);
vectorPerTP = getVector(series); % get wobble vector between timepoints
newCells = [];
for i = 1:max(XLS(:,1))
    cells = XLS(XLS(:,1)==i,:);
    if min(cells(:,3)) > 0
        newCells = [newCells; cells(1,:)]; % get first known coords of all new cells after baseline
    end
end
%% calculate
bslnCellNames = series.tp0(:,1);
lastCellNames = series.(tp{end})(:,1);
bslnIdx = ~ismember(bslnCellNames,lastCellNames);
lastIdx = ~ismember(lastCellNames,bslnCellNames);
bslnStblCoords = series.tp0(~bslnIdx,end-2:end); % stbl only
lastStblCoords = series.(tp{end})(~lastIdx,end-2:end); % stbl only

bslnCells = series.tp0; % !!! NOTE: SUBSEQUENT FUNCTIONS REQUIRE THIS TO BE FINAL bslnCells,lastCells ASSIGNMENTS - DO NOT REARRANGE
lastCells = series.(tp{end});
bslnCoords = bslnCells(:,end-2:end); % all bsln, lost and stbl
lastCoords = lastCells(:,end-2:end); % all last, new and stbl
lastNewCoords = series.(tp{end})(lastIdx,end-2:end); % only include new cells

vectorTot = sum(vectorPerTP(1:end,:),1); % sum vector from baseline through last
[~,Dstbl] = knnsearch(bslnStblCoords + vectorTot, lastStblCoords, 'K', 1);
Dstblnew = NaN;
[~,Dall] = knnsearch(bslnCoords + vectorTot, lastCoords, 'K', 1);

% prove that new cells are stable
if ~isempty(newCells)
    earliest = min(newCells(:,3));
    earlyReplCells = newCells(newCells(:,3)==earliest,:);
    idx = ismember(lastCellNames,earlyReplCells(:,1));
    lastEarly = series.(tp{end})(idx,:);
    vectorTot = sum(vectorPerTP(earliest:end,:),1); % verify this is the correct starting tp in vector matrix
    [~,Dearlynew] = knnsearch(earlyReplCells(:,end-2:end) + vectorTot, lastEarly(:,end-2:end));
else
    Dearlynew = NaN;
end

Dstats.avgDstbl = mean(Dstbl);
Dstats.avgDstblnew = mean(Dstblnew);
Dstats.avgDall = mean(Dall);
Dstats.avgDearlynew = mean(Dearlynew);