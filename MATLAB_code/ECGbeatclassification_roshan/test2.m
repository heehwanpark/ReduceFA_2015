% Pan Tompkins algorithm ver

clc;
clear;

datafolder = 'C:\Users\heehwan\workspace\Data\Standard\mitdb\';
filelist = ['100m'; '101m'; '103m'; '105m'; '106m'; '107m'; '108m'; '109m';
    '111m'; '112m'; '113m'; '114m'; '115m'; '116m'; '117m'; '118m'; '119m'; '121m';
    '122m'; '123m'; '124m'; '200m'; '201m'; '202m'; '203m'; '205m'; '207m'; '208m';
    '209m'; '210m'; '212m'; '213m'; '214m'; '215m'; '217m'; '219m'; '220m'; '221m';
    '222m'; '223m'; '228m'; '230m'; '231m'; '232m'; '233m'; '234m'];

rng(1);

cell_confusion_matrix = traintest_rjmartis_original(datafolder, filelist);
save('results_notsep_pt.mat','cell_confusion_matrix');