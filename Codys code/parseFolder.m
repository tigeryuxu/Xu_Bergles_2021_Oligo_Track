function ALL = parseFolder(cond_dir, ALL, splt, maindir)
%   Parse out time series data and assign to struct for each animal, within
    %either cup/ctrl. Then, take that data and analyze length changes and
    %assign within same animal name in the ALL struct.
    
    
    files = {cond_dir.name};
    %files = files(3:end);
    dirstr = cond_dir.folder;
    cond = strcat(dirstr, '\');
    %cond = dirstr(end-3:end);
%     maindir = 'C:\Users\codyl\OneDrive - Johns Hopkins University\CellTracking\';
    for k = 1:length(files)
        substr = split(files{k},{'_','.'});
        volname = substr{1};
        platform = substr{2};
        obj = substr{3};
        filetype = substr{end};
        if ~contains(filetype,'csv')
            fprintf([volname ' is not a csv file. Convert it.\n']);
            break
        end
        if contains(obj,'10x')
            if contains(platform,'I')
                originalXLS = xlsread(files{k});
                originalXLS(:,end-2:end-1) = originalXLS(:,end-2:end-1) .* 2;
                xyscale = 0.8303;
                zscale = 3;
                dimJ = [1247 1097 198];
                centerJ = dimJ./2;
                centerJ(1) = centerJ(1)+170;
                centerJscaled = [centerJ(1:2).*xyscale, centerJ(3).*zscale];
                originalXLS(:,end-2:end) = (originalXLS(:,end-2:end) - centerJscaled);
                originalXLS(:,3) = originalXLS(:,3) - 1; % first tp is '0' in syGlass
                name = ['v' files{k}(1:3)];
                fprintf('\nParsing volume %s...\n',name);
                ALL.(name).originalXLS = originalXLS;
                if splt
                    ALL = analyzePositions_x4(ALL,name,cond,maindir);
                else
                    ALL = analyzePositions(ALL,name,maindir); %outdated
                end
            else
                name = ['v' files{k}(1:3)];
                fprintf('\nParsing volume %s...\n',name);
                ALL.(name).originalXLS = xlsread(files{k});
                if splt
                    ALL = analyzePositions_x4(ALL,name,cond,maindir);
                else
                    ALL = analyzePositions(ALL,name,maindir); %outdated
                end
            end
        elseif contains(obj,'20x')
            parts = split(volname,'-');
            name = [parts{1}]; % name will be something to define each mouse
            quad = [parts{2}]; % quad is 3# region tag (i.e. r033)
            ALL.(name).(quad).originalXLS = xlsread(files{k});
            ALL = analyzePositions_20(ALL,name,quad,cond,maindir);
        else 
            fprintf(['The objective used for volume ' volname ' is not labeled. Add the objective to the filename.\n']);
            break
        end
    end
end