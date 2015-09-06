% rjmartis main
clc;
clear;

% parameters
datafolder = 'C:\Users\heehwan\workspace\Data\Standard\mitdb\';
filelist = ['100m'; '101m'; '103m'; '105m'; '106m'; '107m'; '108m'; '109m';
    '111m'; '112m'; '113m'; '114m'; '115m'; '116m'; '117m'; '118m'; '119m'; '121m';
    '122m'; '123m'; '124m'; '200m'; '201m'; '202m'; '203m'; '205m'; '207m'; '208m';
    '209m'; '210m'; '212m'; '213m'; '214m'; '215m'; '217m'; '219m'; '220m'; '221m';
    '222m'; '223m'; '228m'; '230m'; '231m'; '232m'; '233m'; '234m'];

rng(1);

[nRecord, ~] = size(filelist);
nTesting = 5;
nTraining = nRecord - nTesting;

results_seg = cell(1,10);
results_beat = cell(1,10);
for i = 1:10
    disp(strcat('Epoch ',int2str(i)))
    
    file_shuffle = randperm(nRecord);
    
    traininglist = filelist(file_shuffle(1:nTraining), :);
    testlist = filelist(file_shuffle(nTraining+1:nRecord), :);
    
    disp('...start training')
    [SVMModel, coeff_cA, coeff_cD] = training_rjmartis_seperated(datafolder, traininglist);
    
    disp('...start testing')
    segment_classification = true;
    confusion_matrix1 = test_rjmartis_seperated(SVMModel, coeff_cA, coeff_cD, datafolder, testlist, segment_classification);
    results_seg{i} = confusion_matrix1;
    disp('----------------')
    segment_classification = false;
    confusion_matrix2 = test_rjmartis_seperated(SVMModel, coeff_cA, coeff_cD, datafolder, testlist, segment_classification);
    results_beat{i} = confusion_matrix2;
end

save('results_seg_pt.mat','results_seg');
save('results_beat_pt.mat','results_beat');

disp('Martis way - 10 fold')
results_notsep = cell(1,10);
confusion_matrix = traintest_rjmartis_original(datafolder, filelist);
results_notsep{i} = confusion_matrix;
save('results_notsep_pt.mat','results_notsep');