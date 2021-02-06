clear variables;
clc;
maindir = uigetdir;
addpath(maindir);
maindir = strcat(maindir, '\');
addpath(pwd)
cd(maindir);
fnames = dir('*.csv');
splt = 1;
ALL=struct;
data = parseFolder(fnames,ALL,splt,maindir);
dt = datestr(datetime('today'),'yyyymmdd');
save(fullfile(maindir,['DATA_notAveraged_' dt '.mat']),'data');