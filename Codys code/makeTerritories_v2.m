function [terrmapB,terrmapL,terrmapLrand] = makeTerritories_v2(directory,cond,name,verticesS,bslnCoords,lastNewCoords,lastRandCoords)
    % baseline territories
    xrSL = 75.7; yrSL = 75.7; zrSL = 32.2; % bsln stable/lost radii
    if contains(cond,{'pr','rB'})
        xrN = 85.25; yrN = 85.25; zrN = 33.75; % remyel ellipsoid radii
    elseif contains(cond,{'rl','lB'})
        xrN = 75.7; yrN = 75.7; zrN = 32.2; % ctrl de novo ellipsoid radii
    end
    tlx = round(verticesS(1,1));
    trx = round(verticesS(2,1));
    tly = round(verticesS(1,2));
    bly = round(verticesS(3,2));
    topz = round(min(verticesS(:,3)));
    botz = round(max(verticesS(:,3)));
    [xm,ym,zm] = meshgrid(tlx:1:trx,tly:1:bly,topz:3:botz);
    terrmapB = zeros(length(tly:1:bly),length(tlx:1:trx),length(topz:3:botz)); %prealloc
    for i = 1:size(bslnCoords,1)
        xc = bslnCoords(i,1); yc = bslnCoords(i,2); zc = bslnCoords(i,3); % ellipse center
        maskd = double( ((xm-xc).^2/(xrSL.^2)) + ((ym-yc).^2/(yrSL.^2)) + ((zm-zc).^2/(zrSL.^2)) <= 1 );
        terrmapB = terrmapB + maskd;
    end
    warning('off','MATLAB:MKDIR:DirectoryExists');
    mkdir(directory,name)
    warning('on','MATLAB:MKDIR:DirectoryExists');
    factor = floor(255/max(max(max(terrmapB,[],3)))); %scale the gray values to fit the entire range
    for i=1:size(terrmapB,3)
        tiff = uint8(factor .* terrmapB(:, :, i));
        outputFileName = [directory '\' name '\' sprintf('bsln%d.tiff', i)];
        imwrite(tiff,outputFileName,'WriteMode','overwrite')
    end
    % last tp territories
    terrmapL = zeros(length(tly:1:bly),length(tlx:1:trx),length(topz:3:botz)); %prealloc
    for i = 1:size(lastNewCoords,1)
        xc = lastNewCoords(i,1); yc = lastNewCoords(i,2); zc = lastNewCoords(i,3); % ellipse center
        maskd = double( ((xm-xc).^2/(xrN.^2)) + ((ym-yc).^2/(yrN.^2)) + ((zm-zc).^2/(zrN.^2)) <= 1 );
        terrmapL = terrmapL + maskd;
    end
    factor = floor(255/max(max(max(terrmapL,[],3))));
    for i=1:size(terrmapL,3)
        tiff = uint8(factor .* terrmapL(:, :, i));
        outputFileName = [directory '\' name '\' sprintf('last%d.tiff', i)];
        imwrite(tiff,outputFileName,'WriteMode', 'overwrite')
    end
%     randomized last tp territories
    terrmapLrand = zeros(length(tly:1:bly),length(tlx:1:trx),length(topz:3:botz)); %prealloc
    if ~isempty(lastRandCoords)
        L = size(lastRandCoords,1);
        xrand=lastRandCoords(:,1);
        yrand=lastRandCoords(:,2);
        zrand=lastRandCoords(:,3);
        for i = 1:L
            xc = xrand(i); yc = yrand(i); zc = zrand(i); % ellipse center
            maskd = double( ((xm-xc).^2/(xrN.^2)) + ((ym-yc).^2/(yrN.^2)) + ((zm-zc).^2/(zrN.^2)) <= 1 );
            terrmapLrand = terrmapLrand + maskd;
        end
    end
    factor = floor(255/max(max(max(terrmapLrand,[],3))));
    for i=1:size(terrmapLrand,3)
        tiff = uint8(factor .* terrmapLrand(:, :, i));
        outputFileName = [directory '\' name '\' sprintf('lastRand%d.tiff', i)];
        imwrite(tiff,outputFileName,'WriteMode', 'overwrite')
    end
end