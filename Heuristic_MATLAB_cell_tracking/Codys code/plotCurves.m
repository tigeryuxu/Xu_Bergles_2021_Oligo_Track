% Compile data by weekly time points based on animal and condition
function [TPdata,WKdata,AlignData,wkIdx,tpAlign] = plotCurves(name,TPdata,skiprest)
% create individual indexing variables for each volume
     %         bsln 1w 2w 3w 3d 5d 1w 10d 2w 3w...
switch name
    case {'v830','v840'}
        tpAlign = {1 NaN NaN 2 NaN NaN 3 NaN NaN NaN NaN 4 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 NaN NaN 2 3 NaN NaN NaN 4 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 NaN NaN 2 3 NaN NaN NaN 4 NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdxCum);
    case {'v610','v620','v650','v680','v690','v700','r710','r720','v820'} % 610cupr+BZA=MOBPF1_013018; 620ctrl+BZA=MOBPM2_013018; 650cupr+BZA=MOBPF_190106w_5; 680cupr+BZA=MOBPF_190105w_1; 690cupr+SHAMgav=MOBPM_190221w_2; 700ctrl+BZA=MOBPM_190226w_3; 710/720cupr+BZA=MOBPM_190226w_2
        tpAlign = {1 2 3 4 NaN NaN 5 NaN 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 2 3 4 5 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdxCum);
    case 'v630' % cupr+BZA=MOBPM2_032018
        tpAlign = {1 2 NaN 3 4 NaN NaN 5 NaN 6 7 NaN NaN NaN NaN 8 NaN NaN NaN NaN};
        wkIdx = {1 2 NaN 3 4 5 6 7 NaN NaN NaN NaN 8 NaN NaN NaN NaN};
        wkIdxCum = {1 2 NaN 3 4 5 6 7 NaN NaN NaN NaN 8 NaN NaN NaN NaN};
        wks = length(wkIdxCum);
    case {'r642','r643','r644'} % ctrl+BZA=MOBPF1_032018
        tpAlign = {1 2 3 4 NaN NaN 5 NaN 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 2 3 4 5 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdxCum); 
    case {'r661','r662','r663','r670','r800','r810'} % 660cupr+BZA=MOBPF_190111w_5; 670cupr+BZA=MOBPF_190112w_8
        tpAlign = {1 NaN 2 3 4 NaN NaN 5 NaN 6 7 8 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 NaN 2 3 4 5 6 7 8 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 NaN 2 3 4 5 6 7 8 NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdxCum);        
    case 'v211' % tp0:1 2wks; tp13:14 2wks ENRICHMENT
        tpAlign = {1 NaN 2 3 4 NaN 5 6 7 8 9 10 11 12 13 14 NaN 15 16 NaN};
        wkIdx = {1 NaN 2 3 5 7 8 9 10 11 12 13 14 NaN 15 16 NaN};
        wkIdxCum = {1 NaN 2 3 4:5 6:7 8 9 10 11 12 13 14 NaN 15 16 NaN};
        wks = length(wkIdxCum);
    case 'v216' % tp0:1 2wks; tp13:14 2wks ENRICHMENT
        tpAlign = {1 NaN 2 3 4 NaN 5 6 7 8 9 10 11 12 13 NaN 14 15 NaN NaN};
        wkIdx = {1 NaN 2 3 5 7 8 9 10 11 12 13 NaN 14 15 NaN NaN};
        wkIdxCum = {1 NaN 2 3 4:5 6:7 8 9 10 11 12 13 NaN 14 15 NaN NaN};
        wks = length(wkIdx);
    case 'v204' % tp0:1 2wks; ENRICHMENT = MOBP 6 82117
        tpAlign = {1 NaN 2 NaN 3 NaN 4 NaN 5 6 7 NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 NaN 2 NaN 4 5 6 7 8 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 NaN 2 NaN 3:4 5 6 7 8 NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdx);
    case 'v260' % tp0:1 2wks; ENRICHMENT
        tpAlign = {1 NaN 2 3 4 NaN 5 6 7 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 NaN 2 3 5 7 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 NaN 2 3 4:5 6:7 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdx);
    case {'r470','r471'} %4/20 cohort mouse2 2 regions 1,2 ENRICHMENT ctrl
        tpAlign = {1 2 3 4 NaN NaN 5 NaN 6 7 8 9 NaN 10 11 NaN NaN NaN NaN NaN};
        wkIdx = {1 2 3 4 5 6 7 8 9 NaN 10 11 NaN NaN NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5 6 7 8 9 NaN 10 11 NaN NaN NaN NaN NaN};
        wks = length(wkIdx);
    case {'r480','r481','r490','r491'} %4/20 cohort mouse1 2 & 3 regions 1,2 ENRICHMENT cupr
        tpAlign = {1 2 3 4 5 NaN 6 NaN 7 8 9 10 NaN 11 12 13 14 15 16 17};
        wkIdx = {1 2 3 4 6 7 8 9 10 NaN 11 12 13 14 15 16 17};
        wkIdxCum = {1 2 3 4 5:6 7 8 9 10 NaN 11 12 13 14 15 16 17};
        wks = length(wkIdx);
    case {'v500'} %180729 cohort mouse1f 10x ENRICHMENT cupr
        tpAlign = {1 2 3 4 NaN NaN 5 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 2 3 4 5 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdx);        
    case {'v510'} %180729 cohort mouse2f 10x ENRICHMENT cupr
        tpAlign = {1 2 3 4 NaN NaN 5 NaN 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 2 3 4 5 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdx);
    case {'v520','v550'} %180729 cohort mouse3f and 2m 10x ENRICHMENT sham
        tpAlign = {1 2 3 4 NaN NaN 5 NaN 6 7 8 9 10 11 12 13 14 NaN NaN NaN};
        wkIdx = {1 2 3 4 5 6 7 8 9 10 11 12 13 14 NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5 6 7 8 9 10 11 12 13 14 NaN NaN NaN};
        wks = length(wkIdx);
    case {'v530'} %180729 cohort mouse4f 10x ENRICHMENT sham
        tpAlign = {1 2 3 4 NaN NaN 5 NaN 6 7 8 9 10 11 12 13 NaN NaN NaN NaN};
        wkIdx = {1 2 3 4 5 6 7 8 9 10 11 12 13 NaN NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5 6 7 8 9 10 11 12 13 NaN NaN NaN NaN};
        wks = length(wkIdx);
    case {'v540',} %180729 cohort mouse1m 10x ENRICHMENT cupr
        tpAlign = {1 2 3 4 NaN NaN 5 NaN 6 7 8 9 NaN 10 11 12 NaN NaN NaN NaN};
        wkIdx = {1 2 3 4 5 6 7 8 9 NaN 10 11 12 NaN NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5 6 7 8 9 NaN 10 11 12 NaN NaN NaN NaN};
        wks = length(wkIdx);

    case 'v277' % tp0:1 2wks; tp13:14 2wks
        tpAlign = {1 NaN 2 3 4 NaN 5 6 7 8 9 10 11 12 13 14 NaN 15 16 NaN};
        wkIdx = {1 NaN 2 3 5 7 8 9 10 11 12 13 14 NaN 15 16 NaN};
        wkIdxCum = {1 NaN 2 3 4:5 6:7 8 9 10 11 12 13 14 NaN 15 16 NaN};
        wks = length(wkIdx);
        
        
        
    %% CUPRIZONE
    case 'v235' % tp0:1 2wks; tp13:14 2wks
        %         bsln 1w 2w 3w 3d 5d 1w 10d 2w 3w...
        tpAlign = {1 NaN 2 3 4 NaN 5 6 7 8 9 10 11 12 13 14 NaN 15 16 NaN};
        wkIdx = {1 NaN 2 3 5 7 8 9 10 11 12 13 14 NaN 15 16 NaN};
        wkIdxCum = {1 NaN 2 3 4:5 6:7 8 9 10 11 12 13 14 NaN 15 16 NaN}; 
        wks = length(wkIdx);
%     case 'r001'
%         tpAlign = {1 2 3 4 5 NaN 6 7 8 9 10 11 12 13 14 NaN NaN NaN NaN NaN};
%         wkIdx = {1 2 3 4 6 7 9 10 11 12 13 14 NaN NaN NaN NaN NaN};
%         wkIdxCum = {1 2 3 4 5:6 7:8 9 10 11 12 13 14 NaN NaN NaN NaN NaN};
%         wks = length(wkIdx);
    case {'r033','r030','r037','r090'}   % 9 weeks
        tpAlign = {1 2 3 4 5 6 7 8 9 10 11 12 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 2 3 4 7 9 10 11 12 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5:7 8:9 10 11 12 NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdx);
        
        
        %%% MAYBE EXCLUDE or substituted???
        
    case 'r097'     % 7 weeks   %%% # 7 is screwed up, substituted with 8
        tpAlign = {1 2 3 4 5 6 7 8 9 10 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 2 3 4 7 9 10 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5:7 8:9 10 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdx);
    case 'r099'    % 6 weeks
        tpAlign = {1 2 3 4 5 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 2 3 4 7 9 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5:7 8:9 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdx);
        
        
        %% CONTROL
    case 'v264' % tp0:1 2wks; tp12:13 2wks  %% subbed in 
        tpAlign = {1 NaN 2 3 4 NaN 5 NaN 6 7 8 9 10 11 12 13 NaN 14 15 NaN};
        wkIdx = {1 NaN 2 3 5 7 8 9 10 11 12 13 NaN 14 15 NaN NaN};
        wkIdxCum = {1 NaN 2 3 4:5 6:7 8 9 10 11 12 13 NaN 14 15 NaN NaN};
        wks = length(wkIdx);
        
    case {'r089','r115','r186'}   % 9 weeks total
        tpAlign = {1 2 NaN 3 NaN NaN 4 5 6 7 8 9 10 NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 2 NaN 3 4 6 7 8 9 10 NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 2 NaN 3 4 5:6 7 8 9 10 NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdx);
        
    case 'r056'  % 6 weeks total
        tpAlign = {1 2 NaN 3 NaN NaN 4 5 6 7 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 2 NaN 3 4 6 7 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 2 NaN 3 4 5:6 7 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdx);
    
    case {'r385','r420'}   % 8 weeks total
        tpAlign = {1 2 3 4 NaN NaN 5 NaN 6 7 8 NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 2 3 4 5 6 7 8 NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5 6 7 8 NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdx);

        
    case 'r339'
        tpAlign = {1 2 3 4 NaN NaN 5 NaN 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 2 3 4 5 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5 6 7 8 9 NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdx);
    case 'r369'
        tpAlign = {1 2 3 4 NaN NaN 5 NaN 6 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdx = {1 2 3 4 5 6 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wkIdxCum = {1 2 3 4 5 6 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN};
        wks = length(wkIdx);

end
if nargin==3 && skiprest
    WKdata = [];
    AlignData = [];
    return
end

numCellsPerTP = TPdata.numCellsPerTP;
numLostPerTP = TPdata.numLostPerTP;
numNewPerTP = TPdata.numNewPerTP;
propRelBsln = TPdata.propRelBsln;
alignCells = NaN(19,1);
alignLost = NaN(19,1);
alignNew = NaN(19,1);
for i = 1:length(tpAlign)
    if isnan(tpAlign{i})
        alignCells(i) = NaN;
        alignLost(i) = NaN;
        alignNew(i) = NaN;
        continue
    end
    alignCells(i) = numCellsPerTP(tpAlign{i});
    alignLost(i) = numLostPerTP(tpAlign{i});
    alignNew(i) = numNewPerTP(tpAlign{i});
end

killCurveAlign = NaN(19,1);
newCurveAlign = NaN(19,1);
if isnan(alignCells(1))
    alignCells(1) = alignCells(3);
end
for i = 1:length(tpAlign)
    if ~isnan(tpAlign{i})
        killCurveAlign(i) = alignCells(1) - sum(alignLost(1:i),'omitnan');
        newCurveAlign(i) = 0 + sum(alignNew(1:i),'omitnan');
    else
        killCurveAlign(i) = NaN;
        newCurveAlign(i) = NaN;
    end
end
%%
numCellsPerWk = NaN(wks,1);
numLostPerWk = NaN(wks,1);
numLost_extra = zeros(wks,1);
numNewPerWk = NaN(wks,1);
numNew_extra = zeros(wks,1);
propRelBslnWk = NaN(wks,1);

for i = 1:length(wkIdx)
    if isnan(wkIdx{i})
        numCellsPerWk(i) = NaN;
        numLostPerWk(i) = NaN;
        numNewPerWk(i) = NaN;
        propRelBslnWk(i) = NaN;
        continue
    end
    % if previous week was not imaged, split number of cells across current
    % and previous weeks to prevent overestimating current week (caveat is
    % that only later recovery time points are linear)
    if i > 2 && isnan(numCellsPerWk(i-1)) 
        divFctr = 2;
        if isnan(numCellsPerWk(i-2))
            divFctr = 3;
            if isnan(numCellsPerWk(i-3))
                divFctr = 4;
            end
        end
    else
        divFctr = 1;
    end
    numCellsPerWk(i) = sum(numCellsPerTP(wkIdx{i}));
    numLostPerWk(i) = sum(numLostPerTP(wkIdxCum{i},1)) ./ divFctr;
    numLost_extra(i) = numLostPerWk(i) .* (divFctr-1); %placeholder for cumulative addition
    numNewPerWk(i) = sum(numNewPerTP(wkIdxCum{i},1)) ./ divFctr;
    numNew_extra(i) = numNewPerWk(i) .* (divFctr-1); %placeholder for cumulative addition
    propRelBslnWk(i) = sum(propRelBsln(wkIdx{i}));
end
% raw numbers
killCurve = NaN(wks,1);
newCurve = NaN(wks,1);
if isnan(numCellsPerWk(1))
    numCellsPerWk(1) = numCellsPerWk(3);
end
for i = 1:wks
    if ~isnan(wkIdx{i})
        killCurve(i) = numCellsPerWk(1) - sum(numLostPerWk(1:i),'omitnan') - numLost_extra(i);
        newCurve(i) = 0 + sum(numNewPerWk(1:i),'omitnan') + numNew_extra(i);
    else
        killCurve(i) = NaN;
        killCurve(i) = NaN;
    end
end
%     figure
%     subplot(1,2,1)
%     plot(0:wks-1,killCurve,'LineWidth',1.5,'Color','m')
%     hold on
%     plot(0:wks-1,newCurve,'LineWidth',1.5,'Color','g')
%     plot(0:wks-1,numCellsPerWk,'LineWidth',1.5,'Color','k')
%     legend({'baseline population','new population','total population'})
%     xlabel('weeks')
%     ylabel('number of cells')
%     ylim([0 max(killCurve)+50])
%     hold off
%     % proportions
%     killCurveProp = killCurve./killCurve(1);
%     newCurveProp = newCurve./killCurve(1);
%     numCellsPerWkProp = numCellsPerWk./numCellsPerWk(1);
%     subplot(1,2,2)
%     plot(0:wks-1,killCurveProp,'LineWidth',1.5,'Color','m')
%     hold on
%     plot(0:wks-1,newCurveProp,'LineWidth',1.5,'Color','g')
%     plot(0:wks-1,numCellsPerWkProp,'LineWidth',1.5,'Color','k')
%     legend({'baseline population','new population','total population'})
%     xlabel('weeks')
%     ylabel('proportion of cells')
%     ylim([0 1])
%     hold off

WKdata = [numCellsPerWk numLostPerWk numNewPerWk killCurve newCurve];
AlignData = [alignCells alignLost alignNew killCurveAlign newCurveAlign];
end