% make mitdb (MIT-BIH Arrhythmia dataset) training and test set for CNN
% each record

clc;
clear;

datafolder = '/media/heehwan/HDD_1TB/WFDB_data/MIT_BIH/';
filelist = ['100m'; '101m'; '102m'; '103m'; '104m'; '105m'; '106m'; '107m'; '108m'; '109m';
                '111m'; '112m'; '113m'; '114m'; '115m'; '116m'; '117m'; '118m'; '119m'; '121m';
                '122m'; '123m'; '124m'; '200m'; '201m'; '202m'; '203m'; '205m'; '207m'; '208m';
                '209m'; '210m'; '212m'; '213m'; '214m'; '215m'; '217m'; '219m'; '220m'; '221m';
                '222m'; '223m'; '228m'; '230m'; '231m'; '232m'; '233m'; '234m'];

n1 = 0;
n2 = 0;

inputs = zeros(947,2500);
targets = zeros(947,1);

for i = 1:48
%     filename = '205m';
    filename = filelist(i,:);
    filenum = filename(1:3);
    
    header = fopen(strcat(datafolder, filename, '.hea'));
    C = textscan(header, '%s %s %s %d %d %d %d %d %s', 2, 'headerLines', 1);
    types = C{9};
    
    isII = 0;
    for j = 1:2
        if strcmp(types{j}, 'MLII')
            isII = 1;
            col_idx = j;
        end
    end
    
    if isII
        structure = load(strcat(datafolder,filename,'.mat'));
        val = structure.val;
        val = val(col_idx,:);
        [~, N] = size(val);

        rhythm_annot = zeros(1,N);

        ann_file = strcat(datafolder,filenum,'.txt');
        fid = fopen(ann_file);
        C = textscan(fid, '%s %d %s %d %d %d %s', -inf, 'headerLines', 1);
        fclose(fid);

        sample_idx = C{2};
        aux = C{7};
        len_aux = length(aux);

        button = 0;

        for j = 1:len_aux
            if button == 0 && strcmp(aux{j}, '(VT')
                button = 1;
                st_idx = sample_idx(j);
            elseif button == 0 && strcmp(aux{j}, '(VFL')
                button = 1;
                st_idx = sample_idx(j);
            elseif button == 0 && strcmp(aux{j}, '(SBR')
                button = 1;
                st_idx = sample_idx(j);
            elseif button == 1 && ~isempty(strfind(aux{j}, '('))
                button = 0;
                end_idx = sample_idx(j)-1;
                n1 = n1+1;
                rhythm_annot(st_idx:end_idx) = 1;
            end
        end

        nSegment = floor(N/360);
        for k = 1:nSegment
            st_idx = 360*(k-1)+1;
            end_idx = st_idx+360*10-1;
            if end_idx <= N && sum(rhythm_annot(st_idx:end_idx)) > 0
                n2 = n2+1;
                input = downsample(interp(val(st_idx:end_idx),25),36);
                input = (input-mean(input))/std(input);
                inputs(n2,:) = input;
                target = 1;
                targets(n2) = target;
            end
        end
    end
end


h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/mitdb_overlapped.h5','/inputs', size(inputs));
h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/mitdb_overlapped.h5','/inputs', inputs);

h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/mitdb_overlapped.h5','/targets', size(targets));
h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/mitdb_overlapped.h5','/targets', targets);

