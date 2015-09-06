% make mitdb (MIT-BIH Arrhythmia dataset) training and test set for CNN
% each record
clc;
clear;

datafolder = 'C:\Users\heehwan\workspace\Data\MIT_BIH\';
filelist = ['100m'; '101m'; '102m'; '103m'; '104m'; '105m'; '106m'; '107m'; '108m'; '109m';
                '111m'; '112m'; '113m'; '114m'; '115m'; '116m'; '117m'; '118m'; '119m'; '121m';
                '122m'; '123m'; '124m'; '200m'; '201m'; '202m'; '203m'; '205m'; '207m'; '208m';
                '209m'; '210m'; '212m'; '213m'; '214m'; '215m'; '217m'; '219m'; '220m'; '221m';
                '222m'; '223m'; '228m'; '230m'; '231m'; '232m'; '233m'; '234m'];

rng(1);
shuffle_idx = randperm(48);
filelist = filelist(shuffle_idx(1:48),:);

training_inputs = zeros(43*180, 3600);
training_targets = zeros(43*180, 1);

testing_inputs = zeros(5*180, 3600);
testing_targets = zeros(5*180, 1);

tr_cursor = 1;
te_cursor = 1;

for i = 1:48
    filename = filelist(i,:);
    filenum = filename(1:3);
    disp(filename);
    
    ecgheader = fopen(strcat(datafolder, filename, '.hea'));
    C = textscan(ecgheader, '%s %s %s %d %d %d %d %d %s', 2, 'headerLines', 1);
    ecgtypes = C{9};
    
    isII = 0;
    for j = 1:2
        if strcmp(ecgtypes{j}, 'MLII')
            isII = 1;
            col_idx = j;
        end
    end
    
    if isII
        ecgfile = load(strcat(datafolder, filename, '.mat')); 
        dataset = ecgfile.val;
        ECG = dataset(col_idx, :);
        ECG = (ECG - mean(ECG))/std(ECG);
        [cood, annot] = getAnnotation(filenum, datafolder);
        
        ann_cursor = 1;
        for k = 1:180
            start_idx = 3600*(k-1)+1;
            end_idx = start_idx+3600-1;
            input = ECG(start_idx:end_idx);
            
            normality = 0;
            while cood(ann_cursor) >= start_idx && cood(ann_cursor) <= end_idx
                if annot(ann_cursor) == 1
                    normality = 1;
                end
                ann_cursor = ann_cursor + 1;
            end
            if i <= 43
                training_inputs(tr_cursor, :) = input;
                training_targets(tr_cursor) = normality;
                tr_cursor = tr_cursor + 1;
            else
                testing_inputs(te_cursor, :) = input;
                testing_targets(te_cursor) = normality;
                te_cursor = te_cursor + 1;
            end
            
        end
    end
end

training_inputs(tr_cursor:end, :) = [];
training_targets(tr_cursor:end) = [];

testing_inputs(te_cursor:end, :) = [];
testing_targets(te_cursor:end) = [];


h5create('C:\Users\heehwan\workspace\Data\mitdb_norm_sep.h5','/training_inputs', [7380 3600]);
h5write('C:\Users\heehwan\workspace\Data\mitdb_norm_sep.h5', '/training_inputs', training_inputs);

h5create('C:\Users\heehwan\workspace\Data\mitdb_norm_sep.h5','/training_targets', [7380 1]);
h5write('C:\Users\heehwan\workspace\Data\mitdb_norm_sep.h5', '/training_targets', training_targets);

h5create('C:\Users\heehwan\workspace\Data\mitdb_norm_sep.h5','/testing_inputs', [900 3600]);
h5write('C:\Users\heehwan\workspace\Data\mitdb_norm_sep.h5', '/testing_inputs', testing_inputs);

h5create('C:\Users\heehwan\workspace\Data\mitdb_norm_sep.h5','/testing_targets', [900 1]);
h5write('C:\Users\heehwan\workspace\Data\mitdb_norm_sep.h5', '/testing_targets', testing_targets);