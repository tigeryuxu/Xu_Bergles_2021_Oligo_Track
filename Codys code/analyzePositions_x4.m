function ALL = analyzePositions_x4(ALL,name,cond,directory)
originalXLS = ALL.(name).originalXLS;
%% Get support points for tetrahedron mask and exclude cells outside of boundaries
% get 8 vertices from imageJ using tp1 from the registered series (input them manually into getVertices)
% get dimensions of image (in microns) from imageJ to calculate the center point (and then use center-relative values 
    % for syGlass coordinate system)
[~,centerJ,verticesJ] = getVertices(name);
xyscale = 0.8303;
zscale = 3;
centerJ(1:2) = centerJ(1:2) .* xyscale;
centerJ(3) = centerJ(3) .* zscale;
verticesJ(:,1:2) = verticesJ(:,1:2) .* xyscale;
verticesJ(:,3) = verticesJ(:,3) .* zscale;
verticesS = verticesJ - centerJ;

[vertsABCD,~,~] = divide10xVol(verticesS);
for vert = 1:4
    verticesS = vertsABCD{vert};
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
    included = ~isnan(tn);
    troubleshootBinnedRepl(cellsInit(included,end-2:end),cellsInit(~included,end-2:end),[],verticesS,name)
    if vert > 1 && vert < 4
        tri = delaunay(vertsABCD{1});
        warning('off','MATLAB:nearlySingularMatrix');
        tn = tsearchn(vertsABCD{1},tri,cellsInit(:,end-2:end));
        warning('on','MATLAB:nearlySingularMatrix');
        included_q1 = ~isnan(tn);
        quadCells = cellsInit(included & ~included_q1,:);
    elseif vert==4
        tri = delaunay(vertsABCD{2});
        tn = tsearchn(vertsABCD{2},tri,cellsInit(:,end-2:end));
        included_q2 = ~isnan(tn);
        tri = delaunay(vertsABCD{3});
        tn = tsearchn(vertsABCD{3},tri,cellsInit(:,end-2:end));
        included_q3 = ~isnan(tn);
        quadCells = cellsInit(included & ~included_q2 & ~included_q3,:);
    else
        includedPrev = ~included;
        quadCells = cellsInit(included & ~includedPrev,:);
    end
    XLS = originalXLS(ismember(originalXLS(:,1),quadCells(:,1)) , :);
    quad = ['quad' sprintf('%d',vert)];
    ALL.(name).(quad).XLS = XLS;
    

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
    [ALL.(name).(quad).TPdata,ALL.(name).(quad).WKdata,ALL.(name).(quad).AlignData] = plotCurves(name,TPdata);

    %% Compare cell body locations and replacement efficacy
    if contains(cond,{'pr','rB'})
        [ALL.(name).(quad).Dstats, ~, ~, ~] = calculateSomaKNN_cupr(name, XLS);
    elseif contains(cond,{'rl','lB'})
        [ALL.(name).(quad).Dstats, ~, ~, ~] = calculateSomaKNN_ctrl(name, XLS);
    end
    %% Analyze everything by depth
    % cells on the border of two bins get assigned to the higher bin (e.g. if cell depth is 200 microns, it goes in bin 2)
    ALL.(name).(quad).binned = parseBins(name, verticesS, XLS, directory, ALL, quad, cond);
end
end
