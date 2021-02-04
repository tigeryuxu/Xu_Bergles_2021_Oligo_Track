% Updated 7/1/18 CLC
function xmlstruct = parseXML_SingleCell(filename)
    tree = xmlread(filename);
    
    xmlstruct = struct('samplespacing',struct([]),'imagesize',struct([]),...
        'paths',[]);
    tracings = tree.getElementsByTagName('tracings').item(0);
    
    xmlstruct.samplespacing = parseAttributes(tracings.getElementsByTagName('samplespacing').item(0));
    xmlstruct.imagesize = parseAttributes(tracings.getElementsByTagName('imagesize').item(0));
    
    % get all path data
    list = tree.getElementsByTagName('path');
    numPaths = list.getLength;
    
    for i=1:numPaths
       xmlstruct.paths(i).attribs = parseAttributes(list.item(i-1)); %since starting with Java, item list starts at 0 in xml file, but matlab list (i) starts at 1
       pnts = list.item(i-1).getElementsByTagName('point');
       numPoints = pnts.getLength;
       pntlist = zeros(5000,6);
       for j=1:numPoints
           pntAttribs = parseAttributes(pnts.item(j-1));
           pnttemp = [str2double(pntAttribs.x) str2double(pntAttribs.xd)... %keep all points so it can be imported back into SNT
               str2double(pntAttribs.y) str2double(pntAttribs.yd) str2double(pntAttribs.z)...
               str2double(pntAttribs.zd)];
           pntlist(j,:) = pnttemp;
       end
       %Check if the scale is in microns, if not, convert coord values
       if i == 1
           if ~(pntlist(1,1) == pntlist(1,2)) && ~(pntlist(1,3) == pntlist(1,4)) && (pntlist(1,5) == pntlist(1,6))
               ismicron = true;
           else 
               ismicron = false;
               fprintf('Traces seem to be in pixels. Convert to microns and recompile...\n');
               %break  %% TIGER - commented out break point
%                if contains(filename, '3_MOBP609-1_Reg1_singleOL-3') && contains(filename, 'd00')
%                    xyscale = 3.8452;
%                elseif contains(filename, '3_MOBP609-1_Reg1_singleOL-3') && ~contains(filename, 'd00')
%                    xyscale = 3.6133;
%                if contains(filename, '4_MOBP609-1_Reg1_singleOL-3')
%                    xyscale = 3.6133;
%                if contains(filename, 'MOBPF08_Reg1_singleOL_15')
%                    xyscale = 7.7084;
%                end
%                width = str2double(xmlstruct.imagesize.width);
%                xmlstruct.samplespacing.x = width ./ xyscale ./ width;
%                xmlstruct.samplespacing.y = xmlstruct.samplespacing.x;
%                zscale = 1;
%                xmlstruct.samplespacing.z = zscale;
           end
       end
       if ismicron
           pntlist(~any(pntlist,2),:) = [];
           xmlstruct.paths(i).points = struct('x',pntlist(:,1),'xd',pntlist(:,2),'y',pntlist(:,3),...
               'yd',pntlist(:,4),'z',pntlist(:,5),'zd',pntlist(:,6));
       else
           pntlist(~any(pntlist,2),:) = [];
           xmlstruct.paths(i).points = struct('x',pntlist(:,1),'xd',(pntlist(:,1).*xmlstruct.samplespacing.x),...
               'y',pntlist(:,3),'yd',(pntlist(:,3).*xmlstruct.samplespacing.y),...
               'z',pntlist(:,5),'zd',(pntlist(:,5).*xmlstruct.samplespacing.z)); 
           for k = 1:length(xmlstruct.paths)
               real_coords = [xmlstruct.paths(k).points.xd,xmlstruct.paths(k).points.yd,xmlstruct.paths(k).points.zd];
               b_real_coords = real_coords(2:end,:);
               a_real_coords = real_coords(1:end-1,:);
               xmlstruct.paths(k).attribs.reallength = sum(sqrt(sum((b_real_coords-a_real_coords).^2,2)));
           end
       end
        span = 25;
        method = 'moving';
        xs = smooth(xmlstruct.paths(i).points.xd, span, method);
        ys = smooth(xmlstruct.paths(i).points.yd, span, method);
        zs = smooth(xmlstruct.paths(i).points.zd, span, method);
        smoothed = [xs,ys,zs];
        xmlstruct.paths(i).points.smoothed = smoothed;
        b_real_coords = smoothed(2:end,:);
        a_real_coords = smoothed(1:end-1,:);
        xmlstruct.paths(i).attribs.reallength_smoothed = sum(sqrt(sum((b_real_coords-a_real_coords).^2,2)));
    end

end %the XML tree is now transformed into a MATLAB structure

% ----- Local function PARSEATTRIBUTES -----
function attributes = parseAttributes(theNode)
% Create attributes structure.

    attributes = [];
    if theNode.hasAttributes
       theAttributes = theNode.getAttributes;
       numAttributes = theAttributes.getLength;
       names = cell(1,numAttributes);
       values = cell(1,numAttributes);
       for count = 1:numAttributes
          attrib = theAttributes.item(count-1);
          names{count} = char(attrib.getName); % preallocated 20180701
          values{count} = char(attrib.getValue);
       end
       
       attributes = cell2struct(values,names,2);
    end
end
