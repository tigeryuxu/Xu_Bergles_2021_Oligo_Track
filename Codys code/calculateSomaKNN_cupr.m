function [Dstats, vectorTot, bslnCoords, bslnLostCoords, lastCoords, lastNewCoords, lastRandCoords] = calculateSomaKNN_cupr(name, XLS, verticesS)
% KNN analysis of baseline vs last tp and compare average distance to baseline vs rotated 90 deg last tp
%% initialize
maxtp = max(XLS(:,3));
% make structure called series with fields for each time point (e.g. tp1, tp2, etc)
for i = 0:maxtp
    str = ['tp' num2str(i)];
    series.(str) = XLS(XLS(:,3) == i, :);
end
tp = fieldnames(series);
% get wobble vector between timepoints
vectorPerTP = getVector(series);
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
%% calculate
bslnCellNames = series.tp0(:,1);
lastCellNames = series.(tp{end})(:,1);
bslnIdx = ~ismember(bslnCellNames,lastCellNames);
lastIdx = ~ismember(lastCellNames,bslnCellNames);
bslnStblCoords = series.tp0(~bslnIdx,end-2:end); % stbl only
lastStblCoords = series.(tp{end})(~lastIdx,end-2:end); % stbl only
vectorTot = sum(vectorPerTP(1:end,:),1); % sum vector from baseline through last

bslnLostCoords = series.tp0(bslnIdx,end-2:end) + vectorTot; % only include cells lost
lastNewCoords = series.(tp{end})(lastIdx,end-2:end); % only include new cells

bslnCells = series.tp0; % !!! NOTE: SUBSEQUENT FUNCTIONS REQUIRE THIS TO BE FINAL bslnCells,lastCells ASSIGNMENTS - DO NOT REARRANGE
% [~,~,~,wkIdx] = plotCurves(name,NaN,1);  EDIT HERE TO MAKE SPECIFIC TIMEPOINT AS LAST ONE
% if tp{end} >
lastCells = series.(tp{end});
bslnCoords = bslnCells(:,end-2:end); % all bsln, lost and stbl
lastCoords = lastCells(:,end-2:end); % all last, new and stbl

[~,Dstbl] = knnsearch(bslnStblCoords + vectorTot, lastStblCoords, 'K', 1);
[~,Dlostnew] = knnsearch(bslnLostCoords, lastNewCoords, 'K', 1);
[~,Dall] = knnsearch(bslnCoords + vectorTot, lastCoords, 'K', 1);

% prove that new cells are stable
if isempty(replCells) 
    Dearlynew = NaN;
    Dstats.avglostnewRand = NaN;
    Dstats.lastnewCoords = [];
    lastRandCoords = [];
else
    earliest = min(replCells(:,3));
    earlyReplCells = replCells(replCells(:,3)==earliest,:);
    idx = ismember(lastCellNames,earlyReplCells(:,1));
    lastEarly = series.(tp{end})(idx,:);
    vectorTot = sum(vectorPerTP(earliest:end,:),1); % verify this is the correct starting tp in vector matrix
    [~,Dearlynew] = knnsearch(earlyReplCells(:,end-2:end) + vectorTot, lastEarly(:,end-2:end));
    %% GENERATE SET OF RANDOM SOMA COORDS, COMPARE TO LOST CELLS (for binned analysis only)
    if nargin>2
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
            if ticker > 2000
                fprintf('resetting xrand, yrand, and ticker \n')
                ticker = 0;
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
            end
        end
        lastRandCoords = [xrand,yrand,zrand];
        [~,DlostnewRand] = knnsearch(bslnLostCoords, lastRandCoords, 'K', 1);
        Dstats.avglostnewRand = mean(DlostnewRand);
        Dstats.randnewCoords = lastRandCoords;
    end
end
%% final assignments
Dstats.avgDstbl = mean(Dstbl);
Dstats.avgDlostnew = mean(Dlostnew);
Dstats.avgDall = mean(Dall);
Dstats.avgDearlynew = mean(Dearlynew);
