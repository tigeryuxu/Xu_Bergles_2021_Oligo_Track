function [stblRepl_SP, stblRepl_NP, SPanalysis, NPanalysis,lastRandCoords] = parseSomaLoc_ctrl(name, XLS, lastNewCoords, verticesS)
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
        newCells = [newCells; cells(1,:)]; % get first known coords of all new cells after the last tp of cupr
    end
end

% DEFINE PARAMETERS FOR VOLUME REPLACEMENT
Rxl = 75.7;
Ryl = 75.7;
Rzl = 32.2;
Vstbl = (4/3).*pi.*Rxl.*Ryl.*Rzl;

Rxr = Rxl;
Ryr = Ryl;
Rzr = Rzl;
Vnew = (4/3).*pi.*Rxr.*Ryr.*Rzr;

thresholdRadius = Rxl + Rxr;

% determine which new cells appear next to stable cells
bslnCells = series.tp0(:,1);
lastCells = series.(tp{end})(:,1);
bslnIdx = ismember(bslnCells,lastCells);
stblNames = series.tp0(bslnIdx,1);
if isempty(newCells) % in case of bin having no new cells across timecourse
    for t = 1:length(tp)
        stblRepl_SP.(tp{t}) = [{NaN} {NaN} {NaN} {NaN} {NaN} {NaN}];
        stblRepl_NP.(tp{t}) = [{NaN} {NaN} {NaN} {NaN} {NaN} {NaN}];
    end
    numCellsRepl = zeros(1,12);
    distCellsRepl = zeros(1,8);
    propCellsRepl = zeros(1,11);
    NPanalysis.numCellsRepl = numCellsRepl;
    NPanalysis.distCellsRepl = distCellsRepl;
    NPanalysis.propCellsRepl = propCellsRepl;
    SPanalysis.numCellsRepl = numCellsRepl;
    SPanalysis.distCellsRepl = distCellsRepl;
    SPanalysis.propCellsRepl = propCellsRepl;
    lastRandCoords = [];
else
    for i = 1:maxtp+1
        stblTPidx = ismember(series.(tp{i})(:,1) , stblNames);
        stblCellsThisTP = series.(tp{i})(stblTPidx,:);
        stblCellsThisTPCoords = stblCellsThisTP(:,end-2:end);
        replCellsThisTP = newCells(newCells(:,3)==i , :);
        replCellsThisTPCoords = replCellsThisTP(:,end-2:end);
        [index,dist] = rangesearch(stblCellsThisTPCoords,replCellsThisTPCoords,thresholdRadius,'Distance','euclidean');
        % VERIFY OVERLAP
        for j = 1:length(index)
            wout = [];
            if isempty(index{j})
                continue
            end
            for w = 1:length(index{j})
                c1 = stblCellsThisTPCoords(index{j}(w), :);
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
                centroid1 = stblCellsThisTPCoords(index{j}(w), :);
                centroid2 = replCellsThisTPCoords(j,:);
                propVolRepl{j}(w) = calcEllipsoidOverlap(centroid1,Rxl,Rzl,centroid2,Rxr,Rzr) ./ Vstbl;
            end
        end
        Repl = cell(size(index,1),3); %prealloc
        for j = 1:length(index)
            Repl{j,1} = stblCellsThisTP(index{j},1); %cell # tags
            Repl{j,2} = stblCellsThisTP(index{j},3); %LKTP of each cell
            Repl{j,3} = (i-1) - Repl{j,2}; %TPs b/w LKTP and TP new cell appeared
        end
        stblRepl_SP.(tp{i}) = [num2cell(replCellsThisTP(:,1)) Repl dist propVolRepl];
        
        [index,dist] = rangesearch(replCellsThisTPCoords,stblCellsThisTPCoords,thresholdRadius,'Distance','euclidean');
        % VERIFY OVERLAP
        for j = 1:length(index)
            wout = [];
            if isempty(index{j})
                continue
            end
            for w = 1:length(index{j})
                c1 = replCellsThisTPCoords(index{j}(w), :);
                c2 = stblCellsThisTPCoords(j,:);
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
                centroid1 = replCellsThisTPCoords(index{j}(w), :);
                centroid2 = stblCellsThisTPCoords(j,:);
                propVolRepl{j}(w) = calcEllipsoidOverlap(centroid1,Rxr,Rzr,centroid2,Rxl,Rzl) ./ Vnew;
            end
        end
        Repl = cell(size(index,1),3); %prealloc
        for j = 1:length(index)
            Repl{j,1} = replCellsThisTP(index{j},1); %cell # tags
            Repl{j,2} = replCellsThisTP(index{j},3); %LKTP of each cell
            Repl{j,3} = (i-1) - Repl{j,2}; %TPs b/w LKTP and TP new cell appeared
        end
        stblRepl_NP.(tp{i}) = [num2cell(stblCellsThisTP(:,1)) Repl dist propVolRepl];
    end
end


%% CALCULATE REPLACEMENT FROM PERPECTIVE OF NEW and LOST CELLS
if ~isempty(newCells)
    for q = 1:2
        if q==1
            lp = stblRepl_NP;
            tp = fieldnames(lp);
            ca = stblRepl_NP.(tp{1});
            for t = 1:length(tp)-1
                for i = 1:size(lp.(tp{t}),1)
                    if isempty(lp.(tp{t}){i,1})
                        continue
                    else
                        for k = 2:4
                            ca{i,k} = [ca{i,k}; lp.(tp{t}){i,k}];
                        end
                        for k = 5:6
                            ca{i,k} = [ca{i,k} lp.(tp{t}){i,k}];
                        end
                    end
                end
            end
        else
            lp = stblRepl_SP;
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
        distCellsRepl = zeros(1,8); %prealloc
        for i = 1:8
            if i==8
                distCellsRepl(1,i) = sum(isnan(lostCellData(:,3)) | isempty(lostCellData(:,3))); % number of cells with no cell within 151.4 micron radius
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
            SPanalysis.numCellsRepl = numCellsRepl;
            SPanalysis.distCellsRepl = distCellsRepl;
            SPanalysis.propCellsRepl = propCellsRepl;
        else
            NPanalysis.numCellsRepl = numCellsRepl;
            NPanalysis.distCellsRepl = distCellsRepl;
            NPanalysis.propCellsRepl = propCellsRepl;
        end
    end
    %% make random last new coords
    tlx = round(verticesS(1,1));
    trx = round(verticesS(2,1));
    tly = round(verticesS(1,2));
    bly = round(verticesS(3,2));
    topz = round(min(verticesS(:,3)));
    botz = round(max(verticesS(:,3)));
    L = size(lastNewCoords,1);
    if contains(name,'v')
        mx = max([bly tly]);
        mn = min([bly tly]);
        if mn==0
            yrange = mx;
        else
            yrange = mx - mn;
        end
        yrand = mn + (yrange)*rand([L 1]);
        mx = max([trx tlx]);
        mn = min([trx tlx]);
        if mn==0
            xrange = mx;
        else
            xrange = mx - mn;
        end
        xrand = mn + (xrange)*rand([L 1]);
    else
        yrand = tly + (bly*2)*rand([L 1]);
        xrand = tlx + (trx*2)*rand([L 1]);
    end
    idx = find(abs(xrand) > max(abs(verticesS(:,1))));
    if sum(idx)
        xrand(idx) = sign(xrand(idx)) .* abs(xrand(idx)) + (abs(xrand(idx)) - max(abs(verticesS(:,1))));
    end
    idx = find(abs(yrand) > max(abs(verticesS(:,2))));
    if sum(idx)
        yrand(idx) = sign(yrand(idx)) .* abs(yrand(idx)) + (abs(yrand(idx)) - max(abs(verticesS(:,2))));
    end
    zrand = [];
    zL = 0;
    ticker = 0;
    while zL < L
        mx = max([botz topz]);
        mn = min([botz topz]);
        zrange = mx - mn;
        ztest = [zrand; mn + (zrange)*rand([L-zL 1])];
        tri = delaunay(verticesS);
        tn = tsearchn(verticesS,tri,[xrand,yrand,ztest]);
        included = ~isnan(tn);
        zrand = ztest(included);
        zL = length(zrand);
        ticker = ticker+1;
        if ticker > 200
            fprintf('halp\n')
        end
    end
    lastRandCoords = [xrand,yrand,zrand];
end
end