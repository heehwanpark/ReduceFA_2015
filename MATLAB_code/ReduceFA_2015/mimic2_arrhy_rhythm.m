% Extract Critical arrhythmia rhythm from MIMIC II ver2 db
% Created by HeeHwan Park

% Data from 'Aboukhalil, Anton, et al. "Reducing false alarm rates for
% critical arrhythmias using the arterial blood pressure waveform." Journal of biomedical informatics 41.3 (2008): 442-451.'

clc;
clear;

h5file = '/home/heehwan/Workspace/Data/ReduceFA_2015/mimic2_savefile_v2.h5';
mimiclist = '/home/heehwan/Workspace/Data/ReduceFA_2015/mimic2_annotation_list_v1';

fid = fopen(mimiclist);
contents = textscan(fid, '%s');
fclose(fid);

filelist = contents{1};
len = length(filelist);

inputs = zeros(100000, 2500);
targets = zeros(100000, 1);

start = 1;
for i = 1:len
    filenum = filelist{i};
    disp(filenum)
    input = h5read(h5file, strcat('/',filenum,'/inputs'));
    input = input';    
    target = h5read(h5file, strcat('/',filenum,'/targets'));
    target = target';
    
    zero_idx = any(input, 2);
    input(zero_idx == 0, :) = [];
    target(zero_idx == 0) = [];
    
    n = length(target);
    resampled_input = zeros(n, 2500);
    for j = 1:n
        resampled_input(j,:) = interp(input(j,:),2);
    end    
    
    inputs(start:start-1+n,:) = resampled_input;
    targets(start:start-1+n) = target;
    start = start + n;
end

inputs(start:end,:) = [];
targets(start:end) = [];

h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/mimic2_overlapped_v2.h5','/inputs', size(inputs));
h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/mimic2_overlapped_v2.h5','/inputs', inputs);

h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/mimic2_overlapped_v2.h5','/targets', size(targets));
h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/mimic2_overlapped_v2.h5','/targets', targets);
