function ALL = analyzePositions_20(ALL,name,quad,cond,directory)
originalXLS = ALL.(name).(quad).originalXLS;
%% Get support points for tetrahedron mask and exclude cells outside of boundaries
% get 8 vertices from imageJ using tp1 from the registered series (input them manually into getVertices)
% get dimensions of image (in microns) from imageJ to calculate the center point (and then use center-relative values
% for syGlass coordinate system)
[~,centerJ,verticesJ] = getVertices(quad);
temp = quad(2:end);
temp = str2double(temp);
if temp > 500
    xyscale = 0.8303;
else
    xyscale = 0.41512723649798663290298476483042;
end
zscale = 3;
centerJ(1:2) = centerJ(1:2) .* xyscale;
centerJ(3) = centerJ(3) .* zscale;
verticesJ(:,1:2) = verticesJ(:,1:2) .* xyscale;
verticesJ(:,3) = verticesJ(:,3) .* zscale;
verticesS = verticesJ - centerJ;
% cells' initial positions are used to fine-tune the boundaries
numCells = max(originalXLS(:,1));
cellsInit = NaN(numCells,size(originalXLS,2));
for i = 1:numCells
    currCell = originalXLS(ismember(originalXLS(:,1),i),:);
    if isempty(currCell)
        continue
    end
    cellsInit(i,:) = currCell(1,:);
end

tri = delaunay(verticesS);
tn = tsearchn(verticesS,tri,cellsInit(:,end-2:end));
excluded = isnan(tn);
includedCellNames = cellsInit(~excluded,1);
XLS = originalXLS(ismember(originalXLS(:,1),includedCellNames) , :);
ALL.(name).(quad).XLS = XLS;

troubleshootBinnedRepl(cellsInit(~excluded,end-2:end),cellsInit(excluded,end-2:end),[],verticesS,quad)
%% Analyze loss and new cells
% Divide cells into matrices by TP
maxtp = max(XLS(:,3));
% make structure called series with fields for each time point (e.g. tp1, tp2, etc)
for i = 0:maxtp
    str = ['tp' num2str(i)];
    series.(str) = XLS(XLS(:,3) == i, :);
end
ALL.(name).(quad).series = series;
% get total number of cells, number lost, number new, per tp
tp = fieldnames(series);
TPdata.numCellsPerTP = NaN(length(tp),1);
TPdata.numNewPerTP = NaN(length(tp),1); % first tp is NaN since unknown how many cells new at baseline
TPdata.numLostPerTP = NaN(length(tp),1); % first tp is NaN since unknown how many cells lost at baseline
TPdata.propRelBsln = NaN(length(tp),1);
for i = 1:length(tp)
    TPdata.numCellsPerTP(i) = length(series.(tp{i})(:,1));
    if i>1
        TPdata.numNewPerTP(i,1) = sum(~ismember(series.(tp{i})(:,1),series.(tp{i-1})(:,1)));
        TPdata.numLostPerTP(i,1) = sum(~ismember(series.(tp{i-1})(:,1),series.(tp{i})(:,1))); % note order of tps
    end
    TPdata.propRelBsln(i) = TPdata.numCellsPerTP(i)./TPdata.numCellsPerTP(1);
end
[ALL.(name).(quad).TPdata, ALL.(name).(quad).WKdata, ALL.(name).(quad).AlignData] = plotCurves(quad,TPdata);
%% Compare cell body locations and replacement efficacy
if contains(cond,{'pr','rB'})
    [ALL.(name).(quad).Dstats, ~, ~, ~] = calculateSomaKNN_cupr(quad, XLS);
elseif contains(cond,{'rl','lB'})
    [ALL.(name).(quad).Dstats, ~, ~, ~] = calculateSomaKNN_ctrl(quad, XLS);
end
%% Analyze everything by depth
% cells on the border of two bins get assigned to the higher bin (e.g. if cell depth is 200 microns, it goes in bin 2)
ALL.(name).(quad).binned = parseBins(quad, verticesS, XLS, directory, ALL, name, cond);
end
