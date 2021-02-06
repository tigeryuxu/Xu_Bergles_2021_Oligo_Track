function [lostRepl_RP, lostRepl_RPrand, lostRepl_LP, RPanalysis, RPrandanalysis, LPanalysis, lostAnyRepl, stblRepl] = parseSomaLoc_cupr(name, XLS, lastRandCoords)
maxtp = max(XLS(:,3));
% make structure called series with fields for each time point (e.g. tp1, tp2, etc)
for i = 0:maxtp
    str = ['tp' num2str(i)];
    series.(str) = XLS(XLS(:,3) == i, :);
end
tp = fieldnames(series);
vectorPerTP = getVector(series); % get wobble vector between timepoints
% get last time point of cupr or sham
[~,~,~,~,tpAlign] = plotCurves(name,[],1);
tpcheck = sum(isnan([tpAlign{1:4}]));
if tpcheck==0
    lastCuprTP = 3;
elseif tpcheck==1 && isnan(tpAlign{4}) && ~isnan(tpAlign{3})
    lastCuprTP = 2;
else
    lastCuprTP = tpAlign{4} - 1;
end
% if contains(name,'204')
%     lastCuprTP = 2;
% elseif contains(name,'v')
%     lastCuprTP = 3;
% elseif contains(name,'r')
%     lastCuprTP = 4;
% end
lostCells = [];
replCells = [];
for i = 1:max(XLS(:,1))
    cells = XLS(XLS(:,1)==i,:);
    if max(cells(:,3)) < maxtp
        lostCells = [lostCells; cells(end,:)]; % get last known coords of all lost cells
    end
    if min(cells(:,3)) > lastCuprTP
        replCells = [replCells; cells(1,:)]; % get first known coords of all new cells after the last tp of cupr
    end
end
% DEFINE PARAMETERS FOR VOLUME REPLACEMENT
Rxl = 75.7;
Ryl = 75.7;
Rzl = 32.2;
Vlost = (4/3).*pi.*Rxl.*Ryl.*Rzl;

Rxr = 85.25;
Ryr = 85.25;
Rzr = 33.75;
Vrepl = (4/3).*pi.*Rxr.*Ryr.*Rzr;

thresholdRadius = Rxl + Rxr;


% bslnCellCoords = series.tp0(:,end-2:end) + sum(vectorPerTP,1);  % USE IF WANT TO INCLUDE STABLE CELLS, SWITCH WITH 'lostCellCoords'
lostCellCoords = NaN(size(lostCells(:,4:end)));
for t = 1:size(lostCells,1)
    lastTP = lostCells(t,3);
    lostCellCoords(t,:) = lostCells(t,4:end) + sum(vectorPerTP(lastTP+1:end,:),1);
end
[index,dist] = rangesearch(lostCellCoords,lastRandCoords,thresholdRadius,'Distance','euclidean');
% VERIFY OVERLAP
for j = 1:length(index)
    wout = [];
    if isempty(index{j})
        continue
    end
    for w = 1:length(index{j})
        c1 = lostCellCoords(index{j}(w), :);
        c2 = lastRandCoords(j,:);
        d = c1-c2;
        in = (d(1)./ (Rxr+Rxl)).^2 + (d(2) ./ (Rxr+Rxl)).^2 + (d(3) ./ (Rzr+Rzl)).^2 <= 1; %check whether the cells' ellipsoids actually overlap
        if ~in
            wout = [wout w];
        end
    end
    index{j}(wout) = [];
    dist{j}(wout) = [];
end
% calculate volume replaced
propVolRepl = cell(length(dist),1);
for j = 1:size(index,1)
    for w = 1:length(index{j})
        if isempty(index{j})
            propVolRepl{j}(w) = NaN;
            continue
        else
            centroid1 = lostCellCoords(index{j}(w), :);
            centroid2 = lastRandCoords(j,:);
            propVolRepl{j}(w) = calcEllipsoidOverlap(centroid1,Rxl,Rzl,centroid2,Rxr,Rzr) ./ Vlost;
        end
    end
end
Repl = cell(size(index,1),3); %prealloc
for j = 1:length(index)
    Repl{j,1} = lostCellCoords(index{j},1); %cell # tags
    Repl{j,2} = NaN; %LKTP of each cell
    Repl{j,3} = NaN; %TPs b/w LKTP and TP new cell appeared
end
lostRepl_RPrand = [num2cell(1:size(lastRandCoords,1))' Repl dist propVolRepl];

%% Real data comparison
for i = lastCuprTP+1:maxtp+1
    %% From perspective of new cells
    if isempty(lostCells)
        continue
    end
    lostCellsThisTPandPrev = series.tp0;    % lostCells(lostCells(:,3)<=i , :); CURRENTLY COMPARING NEW CELLS TO ALL CELLS AT BASELINE, AS IN CONTROL
    lostCellsThisTPandPrev = sortrows(lostCellsThisTPandPrev,3);
    lostCellsThisTPandPrevCoords = NaN(size(lostCellsThisTPandPrev,1) , 3); %prealloc
    for k = 0:i-1
        kidx = lostCellsThisTPandPrev(:,3)==k;
        lostCellsAtTPk = lostCellsThisTPandPrev(kidx , :);
        lostCellsThisTPandPrevCoords(kidx,:) = lostCellsAtTPk(:,end-2:end) + sum(vectorPerTP(k+1:i-1,:),1); % add vectors
    end
    replCellsThisTP = replCells(replCells(:,3)==i , :);
    replCellsThisTPCoords = replCellsThisTP(:,end-2:end);
    [index,dist] = rangesearch(lostCellsThisTPandPrevCoords,replCellsThisTPCoords,thresholdRadius,'Distance','euclidean');
    % VERIFY OVERLAP
    for j = 1:length(index)
        wout = [];
        if isempty(index{j})
            continue
        end
        for w = 1:length(index{j})
            c1 = lostCellsThisTPandPrevCoords(index{j}(w), :);
            c2 = replCellsThisTPCoords(j,:);
            d = c1-c2;
            in = (d(1)./ (Rxr+Rxl)).^2 + (d(2) ./ (Rxr+Rxl)).^2 + (d(3) ./ (Rzr+Rzl)).^2 <= 1; %check whether the cells' ellipsoids actually overlap
            if ~in
                wout = [wout w];
            end
        end
        index{j}(wout) = [];
        dist{j}(wout) = [];
    end
    % calculate volume replaced
    propVolRepl = cell(length(dist),1);
    for j = 1:size(index,1)
        for w = 1:length(index{j})
            if isempty(index{j})
                propVolRepl{j}(w) = NaN;
                continue
            else
                centroid1 = lostCellsThisTPandPrevCoords(index{j}(w), :);
                centroid2 = replCellsThisTPCoords(j,:);
                propVolRepl{j}(w) = calcEllipsoidOverlap(centroid1,Rxl,Rzl,centroid2,Rxr,Rzr) ./ Vlost;
            end
        end
    end
    Repl = cell(size(index,1),3); %prealloc
    for j = 1:length(index)
        Repl{j,1} = lostCellsThisTPandPrev(index{j},1); %cell # tags
        Repl{j,2} = lostCellsThisTPandPrev(index{j},3); %LKTP of each cell
        Repl{j,3} = (i-1) - Repl{j,2}; %TPs b/w LKTP and TP new cell appeared
    end
    lostRepl_RP.(tp{i}) = [num2cell(replCellsThisTP(:,1)) Repl dist propVolRepl];
    
    %% From perspective of lost cells
    [indexLP,distLP] = rangesearch(replCellsThisTPCoords,lostCellsThisTPandPrevCoords,thresholdRadius,'Distance','euclidean');
    % VERIFY OVERLAP
    for j = 1:length(indexLP)
        wout = [];
        if isempty(indexLP{j})
            continue
        end
        for w = 1:length(indexLP{j})
            c1 = replCellsThisTPCoords(indexLP{j}(w), :);
            c2 = lostCellsThisTPandPrevCoords(j,:);
            d = c1-c2;
            in = (d(1)./ (Rxr+Rxl)).^2 + (d(2) ./ (Rxr+Rxl)).^2 + (d(3) ./ (Rzr+Rzl)).^2 <= 1; %check whether the cells' ellipsoids actually overlap
            if ~in
                wout = [wout w];
            end
        end
        indexLP{j}(wout) = [];
        distLP{j}(wout) = [];
    end
    % calculate volume replaced 
    propVolReplLP = cell(length(distLP),1); %prealloc
    for j = 1:length(indexLP)
        for w = 1:length(indexLP{j})
            if isempty(indexLP{j})
                propVolReplLP{j}(w) = NaN;
                continue
            else
                centroid1 = replCellsThisTPCoords(indexLP{j}(w), :);
                centroid2 = lostCellsThisTPandPrevCoords(j,:);
                propVolReplLP{j}(w) = calcEllipsoidOverlap(centroid1,Rxr,Rzr,centroid2,Rxl,Rzl) ./ Vrepl;
            end
        end
    end
    Repl = cell(size(indexLP,1),3); %prealloc
    for j = 1:length(indexLP)
        Repl{j,1} = replCellsThisTP(indexLP{j},1); %cell # tags
        Repl{j,2} = replCellsThisTP(indexLP{j},3); %LKTP of each cell
        Repl{j,3} = (i-1) - Repl{j,2}; %TPs b/w LKTP and TP new cell appeared
    end
    lostRepl_LP.(tp{i}) = [num2cell(lostCellsThisTPandPrev(:,1)) Repl distLP propVolReplLP];
    
    % include cells that die in the future
    [index,dist] = rangesearch(lostCells(:,end-2:end),replCellsThisTPCoords,thresholdRadius,'Distance','euclidean');
    % VERIFY OVERLAP
    for j = 1:length(index)
        wout = [];
        if isempty(index{j})
            continue
        end
        for w = 1:length(index{j})
            c1 = lostCells(index{j}(w), end-2:end);
            c2 = replCellsThisTPCoords(j,:);
            d = c1-c2;
            in = (d(1)./ (Rxr+Rxl)).^2 + (d(2) ./ (Rxr+Rxl)).^2 + (d(3) ./ (Rzr+Rzl)).^2 <= 1; %check whether the cells' ellipsoids actually overlap
            if ~in
                wout = [wout w];
            end
        end
        index{j}(wout) = [];
        dist{j}(wout) = [];
    end
    % calculate volume replaced
    propVolRepl = cell(length(dist),1);
    for j = 1:length(index)
        for w = 1:length(index{j})
            if isempty(index{j})
                propVolRepl{j}(w) = NaN;
                continue
            else
                centroid1 = lostCells(index{j}(w), end-2:end);
                centroid2 = replCellsThisTPCoords(j,:);
                propVolRepl{j}(w) = calcEllipsoidOverlap(centroid1,Rxr,Rzr,centroid2,Rxl,Rzl) ./ Vlost;
            end
        end
    end
    Repl = cell(size(index,1),3); %prealloc
    for j = 1:length(index)
        Repl{j,1} = lostCells(index{j},1); %cell # tags
        Repl{j,2} = lostCells(index{j},3); %LKTP of each cell
        Repl{j,3} = (i-1) - Repl{j,2}; %TPs b/w LKTP and TP new cell appeared
    end
    lostAnyRepl.(tp{i}) = [num2cell(replCellsThisTP(:,1)) Repl dist propVolRepl];
end

% determine which new cells appear next to stable cells
bslnCells = series.tp0(:,1);
lastCells = series.(tp{end})(:,1);
bslnIdx = ismember(bslnCells,lastCells);
stblNames = series.tp0(bslnIdx,1);
for i = lastCuprTP+1:maxtp+1
    stblTPidx = ismember(series.(tp{i})(:,1) , stblNames);
    stblCellsThisTP = series.(tp{i})(stblTPidx,:);
    replCellsThisTP = replCells(replCells(:,3)==i , :);
    [index,dist] = rangesearch(stblCellsThisTP(:,end-2:end),replCellsThisTP(:,end-2:end),thresholdRadius,'Distance','euclidean');
    Repl = cell(size(index,1),1); %prealloc
    for j = 1:length(index)
        Repl{j,1} = stblCellsThisTP(index{j},1); %cell # tags
    end
    stblRepl.(tp{i}) = [num2cell(replCellsThisTP(:,1)) Repl dist];
end

%previous location of KNN analysis
%% CALCULATE REPLACEMENT FROM PERPECTIVE OF NEW and LOST CELLS
for q = 1:3
    if q==1
        lp = lostRepl_LP;
        tp = fieldnames(lp);
        ca = lp.(tp{end});
        for t = 1:length(tp)-1
            for i = 1:size(lp.(tp{t}),1)
                for k = 2:4
                    ca{i,k} = [ca{i,k}; lp.(tp{t}){i,k}];
                end
                for k = 5:6
                    ca{i,k} = [ca{i,k} lp.(tp{t}){i,k}];
                end
            end
        end
    elseif q == 2
        ca = lostRepl_RPrand;
    elseif q == 3
        lp = lostRepl_RP;
        tp = fieldnames(lp);
        ca = [];
        for t = 1:length(tp)
            ca = [ca; lp.(tp{t})];
        end
    end

    lostCellData = zeros(size(ca,1),4);
    for i = 1:size(ca,1)
        lostCellData(:,1) = cell2mat(ca(:,1));
        lostCellData(i,2) = length(ca{i,2});
        lostCellData(i,3) = mean(ca{i,5});
        if q == 1
            lostCellData(i,4) = sum(ca{i,6});
        else
            lostCellData(i,4) = mean(ca{i,6});
        end
    end
    numCellsRepl = zeros(1,12); %prealloc
    for i = 0:11
        if i == 11
            numCellsRepl(1,i+1) = sum(lostCellData(:,2)>10);
        else
            numCellsRepl(1,i+1) = sum(lostCellData(:,2)==i);
        end
    end
    b = 0:20:thresholdRadius;
    distCellsRepl = zeros(1,9); %prealloc
    for i = 1:9
        if i==9
            distCellsRepl(1,i) = sum(isnan(lostCellData(:,3)) | isempty(lostCellData(:,3))); % number of cells with no cell within 161.95 micron radius
        else
            distCellsRepl(1,i) = sum(lostCellData(:,3)>=b(i) & lostCellData(:,3)<b(i+1));
        end
    end
    propCellsRepl = zeros(1,11); %prealloc
    for i = 0:10
        if i == 10
            propCellsRepl(1,i+1) = sum(lostCellData(:,4)>1);
        else
            propCellsRepl(1,i+1) = sum(lostCellData(:,4)>=i/10 & lostCellData(:,4)<(i+1)/10);
        end
    end
    if q == 1
        LPanalysis.numCellsRepl = numCellsRepl;
        LPanalysis.distCellsRepl = distCellsRepl;
        LPanalysis.propCellsRepl = propCellsRepl;
    elseif q == 2
        RPrandanalysis.numCellsRepl = numCellsRepl;
        RPrandanalysis.distCellsRepl = distCellsRepl;
        RPrandanalysis.propCellsRepl = propCellsRepl;
    elseif q == 3
        RPanalysis.numCellsRepl = numCellsRepl;
        RPanalysis.distCellsRepl = distCellsRepl;
        RPanalysis.propCellsRepl = propCellsRepl;
    end
end
end