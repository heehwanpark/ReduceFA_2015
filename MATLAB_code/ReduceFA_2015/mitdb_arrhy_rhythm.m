% make mitdb (MIT-BIH Arrhythmia dataset) training and test set for CNN
% each record
clc;
clear;

datafolder = '/media/salab-heehwan/HDD_1TB/WFDB_data/MIT_BIH/';
filelist = ['100m'; '101m'; '102m'; '103m'; '104m'; '105m'; '106m'; '107m'; '108m'; '109m';
                '111m'; '112m'; '113m'; '114m'; '115m'; '116m'; '117m'; '118m'; '119m'; '121m';
                '122m'; '123m'; '124m'; '200m'; '201m'; '202m'; '203m'; '205m'; '207m'; '208m';
                '209m'; '210m'; '212m'; '213m'; '214m'; '215m'; '217m'; '219m'; '220m'; '221m';
                '222m'; '223m'; '228m'; '230m'; '231m'; '232m'; '233m'; '234m'];

for i = 1:1
    filename = '205m';
%     filename = filelist(i,:);
    filenum = filename(1:3);
    disp(filename);
    
    structure = load(strcat(datafolder,filename,'.mat'));
    val = structure.val;
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
        elseif button == 1 && strcmp(aux{j}, '(N')
            button = 0;
            end_idx = sample_idx(j)-1;
            
            rhythm_annot(st_idx:end_idx) = 1;
        end
    end
    new_val = [val;rhythm_annot];
end
