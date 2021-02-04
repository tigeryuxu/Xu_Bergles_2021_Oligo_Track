% analyze loss and new cells in 100 micron bins
function [binned] = parseBins(name, verticesS, XLS, directory, ~, quad, cond)
if nargin==2 % to maintain compatability with full 10x code temporarily
    XLS = verticesS;
    verticesS = name;
end
BinnedAlign = struct;
% 0-100 microns
bin1lowerZ = [verticesS(1:4,1:2) verticesS(1:4,3) + 100];
binnedverts.b1 = [verticesS(1:4,:); bin1lowerZ];
binnedvertsRepl.b1 = [verticesS(1:4,:); bin1lowerZ];
% 100-200 microns
bin2upperZ = bin1lowerZ;
bin2lowerZ = bin1lowerZ + [0 0 100];
binnedverts.b2 = [bin2upperZ; bin2lowerZ];
binnedvertsRepl.b2 = [bin2upperZ; bin2lowerZ];
% 200-300 microns
bin3upperZ = bin2lowerZ;
bin3lowerZ = bin2lowerZ + [0 0 100];
binnedverts.b3 = [bin3upperZ; bin3lowerZ];
binnedvertsRepl.b3 = [bin3upperZ; bin3lowerZ];

maxtp = max(XLS(:,3));
cellNames = unique(XLS(:,1));
numCells = length(cellNames);
cellsInit = NaN(numCells,size(XLS,2));
for i = 1:numCells
    currCell = XLS(ismember(XLS(:,1),cellNames(i)),:);
    cellsInit(i,:) = currCell(1,:);
end
b = {'b1','b2','b3'};
mesh = alphaShape(verticesS);
figure
plot(mesh,'FaceColor','k','FaceAlpha',0.05,'EdgeColor','none');
hold on
title([name quad])
plot3(cellsInit(:,end-2),cellsInit(:,end-1),cellsInit(:,end),'ko');
color = {'r','g','b'};
for i = 1:3
    tri = delaunay(binnedverts.(b{i}));
    tn = tsearchn(binnedverts.(b{i}),tri,cellsInit(:,end-2:end));
    included = ~isnan(tn);
    if i > 1
        tri = delaunay(binnedverts.(b{i-1}));
        tn = tsearchn(binnedverts.(b{i-1}),tri,cellsInit(:,end-2:end));
        includedPrev = ~isnan(tn);
    else
        includedPrev = ~included;
    end
    binnedCells.(b{i}) = cellsInit(included & ~includedPrev,:);
    binnedXLS.(b{i}) = XLS(ismember(XLS(:,1),binnedCells.(b{i})(:,1)) , :);
    mesh2 = alphaShape(binnedverts.(b{i}));
    plot(mesh2,'FaceColor',color{i},'FaceAlpha',0.08,'EdgeColor','none');
    plot3(binnedCells.(b{i})(:,end-2),binnedCells.(b{i})(:,end-1),binnedCells.(b{i})(:,end),[color{i} '.']);
    % make structure called series with fields for each time point (e.g. tp1, tp2, etc)
    for k = 0:maxtp
        str = ['tp' num2str(k)];
        binnedseries.(b{i}).(str) = binnedXLS.(b{i})(binnedXLS.(b{i})(:,3)==k, :);
    end
    % get total number of cells, number lost, number new, per tp
    tp = fieldnames(binnedseries.(b{i}));
    binnedData.(b{i}).numCellsPerTP = NaN(length(tp),1);
    binnedData.(b{i}).numNewPerTP = NaN(length(tp),1); % first tp is NaN since unknown how many cells new at baseline
    binnedData.(b{i}).numLostPerTP = NaN(length(tp),1); % first tp is NaN since unknown how many cells lost at baseline
    binnedData.(b{i}).propRelBsln = NaN(length(tp),1);
    for k = 1:length(tp)
        binnedData.(b{i}).numCellsPerTP(k) = length(binnedseries.(b{i}).(tp{k})(:,1));
        if k>1
            binnedData.(b{i}).numNewPerTP(k) = sum(~ismember(binnedseries.(b{i}).(tp{k})(:,1),binnedseries.(b{i}).(tp{k-1})(:,1)));
            binnedData.(b{i}).numLostPerTP(k) = sum(~ismember(binnedseries.(b{i}).(tp{k-1})(:,1),binnedseries.(b{i}).(tp{k})(:,1))); % note order of tps
        end
        binnedData.(b{i}).propRelBsln(k) = binnedData.(b{i}).numCellsPerTP(k)./binnedData.(b{i}).numCellsPerTP(1);
    end
    [~,WKdata,AlignData] = plotCurves(name,binnedData.(b{i}));
    L = size(AlignData,1);
    if L < 18
        AlignData = [AlignData; NaN(18-L,5)]; %#ok<AGROW>
        WKdata = [WKdata; NaN(18-L,5)]; %#ok<AGROW>
    end
    BinnedAlign.(b{i}) = AlignData;
    BinnedWKdata.(b{i}) = WKdata;
    
    %% PARSE SOMA KNN AND VOLUME REPLACEMENT USING BUFFERED VOLUMES
    cellsBsln = binnedCells.(b{i})(binnedCells.(b{i})(:,3)==0 , :);
    cellsInitNew = cellsInit(~cellsInit(:,3)==0 , :);
    tri = delaunay(binnedvertsRepl.(b{i}));
    tn = tsearchn(binnedvertsRepl.(b{i}),tri,cellsInitNew(:,end-2:end));
    included = ~isnan(tn);
    binnedCellsRepl.(b{i}) = [cellsBsln; cellsInitNew(included,:)];
    %         troubleshootBinnedRepl(cellsBsln(:,end-2:end), cellsInitNew(included,end-2:end), binnedvertsRepl.(b{i}), binnedverts.(b{i}))
    binnedXLSrepl.(b{i}) = XLS(ismember(XLS(:,1),binnedCellsRepl.(b{i})(:,1)) , :);
    if i==1
        if contains(cond,{'pr','rB'}) % quick fix for enrichment functionality, where cond == 'prEn'
            [binned.Dstats.(b{i}), vectorTot, ~, bslnLostCoords, ~, lastNewCoords, lastRandCoords] = calculateSomaKNN_cupr(name, binnedXLSrepl.(b{i}), binnedverts.(b{i}));
            [binned.lostRepl_RP, binned.lostRepl_RPrand, binned.lostRepl_LP, binned.RPanalysis, binned.RPrandanalysis, binned.LPanalysis,...
                binned.lostAnyRepl, binned.stblRepl] = parseSomaLoc_cupr(name, binnedXLSrepl.(b{i}), lastRandCoords);
            bslnCoords = bslnLostCoords; % territory comparisons will be made only based off of lost cells
        elseif contains(cond,{'rl','lB'})
            [binned.Dstats.(b{i}), vectorTot, bslnCoords, ~, lastNewCoords] = calculateSomaKNN_ctrl(name, binnedXLSrepl.(b{i}));
            [binned.stblRepl_SP, binned.stblRepl_NP, binned.SPanalysis, binned.NPanalysis,lastRandCoords]...
                = parseSomaLoc_ctrl(name, binnedXLSrepl.(b{i}),lastNewCoords,binnedverts.(b{i}));
            % territory comparisons will compare all baseline cells to new cells
        end
        %% CALCULATE TERRITORY DENSITY CORRELATION
        if exist([directory name quad '\' b{i} '\' 'terrmapB.mat'],'dir')
            load([directory name quad '\' b{i} '\' 'terrmapB.mat'], 'terrmapB');
            load([directory name quad '\' b{i} '\' 'terrmapL.mat'], 'terrmapL');
            load([directory name quad '\' b{i} '\' 'terrmapLrand.mat'], 'terrmapLrand');
        else
            [terrmapB,terrmapL,terrmapLrand]...
                = makeTerritories_v2(directory,cond,[name quad '\' b{i}],binnedverts.(b{i}),bslnCoords+vectorTot,lastNewCoords,lastRandCoords); % currently excluding stable cells in cupr-treated conditions, see line 97
            save([directory name quad '\' b{i} '\' 'terrmapB.mat'],'terrmapB');
            save([directory name quad '\' b{i} '\' 'terrmapL.mat'],'terrmapL');
            save([directory name quad '\' b{i} '\' 'terrmapLrand.mat'],'terrmapLrand');
        end
        binned.terrmapB.(b{i}) = terrmapB;
        binned.terrmapL.(b{i}) = terrmapL;
        binned.terrmapLrand.(b{i}) = terrmapLrand;
        %             binned.terrStats.blTerrR.(b{i}) = corrcoef(binned.terrmapB.(b{i}), binned.terrmapL.(b{i}));
        %             binned.terrStats.blrandTerrR.(b{i}) = corrcoef(binned.terrmapB.(b{i}), binned.terrmapLrand.(b{i}));
        %             binned.terrStats.LvsLrandTerrR.(b{i}) = corrcoef(binned.terrmapL.(b{i}), binned.terrmapLrand.(b{i}));
        
        LvB = terrmapL - terrmapB;
        emptyIdx = (terrmapL==0) & (terrmapB==0);
        LvB(emptyIdx)=NaN;
        totvolB = sum(terrmapB(:));
        totvolL = sum(terrmapL(:));
        novlvol = sum(LvB(LvB > 0),'omitnan');
        temp = LvB;
        temp(temp>0) = 0;
        extrareplvol = temp + terrmapB;
        replvol = sum(extrareplvol(:),'omitnan');
        lostvol = -1 * sum(LvB(LvB < 0), 'omitnan');
        binned.terrStats.vol = [size(lastNewCoords,1), replvol/totvolB, lostvol/totvolB, novlvol/totvolB, replvol/totvolL, novlvol/totvolL];
        
        %rand
        LrandvB = terrmapLrand - terrmapB;
        emptyIdx = (terrmapLrand==0) & (terrmapB==0);
        LrandvB(emptyIdx)=NaN;
        novlvol = sum(LrandvB(LrandvB > 0),'omitnan');
        temp = LrandvB;
        temp(temp>0) = 0;
        extrareplvol = temp + terrmapB;
        replvol = sum(extrareplvol(:),'omitnan');
        lostvol = -1 * sum(LrandvB(LrandvB < 0), 'omitnan');
        binned.terrStats.randvol = [size(lastNewCoords,1), replvol/totvolB, lostvol/totvolB, novlvol/totvolB, replvol/totvolL, novlvol/totvolL];
    else
        if contains(cond,{'pr','rB'})
            [binned.Dstats.(b{i}), vectorTot, ~, bslnLostCoords, ~, lastNewCoords, ~] = calculateSomaKNN_cupr(name, binnedXLSrepl.(b{i}), binnedverts.(b{i}));
            bslnCoords = bslnLostCoords;
        elseif contains(cond,{'rl','lB'})
            [binned.Dstats.(b{i}), vectorTot, bslnCoords, ~, lastNewCoords] = calculateSomaKNN_ctrl(name, binnedXLSrepl.(b{i}));
        end
    end
end
hold off
view([0 5])
check = length(unique(binnedCells.b1(:,1))) + length(unique(binnedCells.b2(:,1))) + length(unique(binnedCells.b3(:,1)));
if ~(check == numCells)
    warning 'The total number of cells across bins is not equal to the original number of cells.'
    fprintf('Total binned = %d\n',check);
    fprintf('Total original = %d\n',numCells);
end
binned.data = binnedData;
binned.series = binnedseries;
binned.cells = binnedCells;
binned.align = BinnedAlign;
binned.wkdata = BinnedWKdata;
end