function troubleshootBinnedRepl(cellsA,cellsB,binnedvertsRepl,verts,name)
if isempty(binnedvertsRepl) %cellsA included, cellsB excluded
    mesh = alphaShape(verts);
    figure
    plot(mesh,'FaceColor','k','FaceAlpha',0.1,'EdgeColor','none');
    title(name)
    hold on
    plot3(cellsA(:,1),cellsA(:,2),cellsA(:,3),'go');
    plot3(cellsB(:,1),cellsB(:,2),cellsB(:,3),'ro');
    hold off
    view([0 5])
else
    mesh = alphaShape(verts);
    figure
    plot(mesh,'FaceColor','k','FaceAlpha',0.1,'EdgeColor','none');
    hold on
    plot3(cellsA(:,1),cellsA(:,2),cellsA(:,3),'ko');
    mesh2 = alphaShape(binnedvertsRepl);
    plot(mesh2,'FaceColor','g','FaceAlpha',0.08,'EdgeColor','none');
    plot3(cellsB(:,1),cellsB(:,2),cellsB(:,3),'go');
    hold off
    view([0 5])
end