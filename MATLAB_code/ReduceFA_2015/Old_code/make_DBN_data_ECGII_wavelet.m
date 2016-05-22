% Change DBN ECGII dataset by using wavelet

clc;
clear;

load('DBN_data\data_10sec_dbn_ECGII_pretraining.mat');
load('DBN_data\data_10sec_dbn_ECGII_training.mat');

[pm, pn] = size(pretraining);
[tm, tn] = size(training);

new_predata_swa = zeros(pm,pn);
new_predata_swd = zeros(pm,pn);
new_trdata_swa = zeros(tm,tn);
new_trdata_swd = zeros(tm,tn);

for i = 1:pm
    [swa, swd] = swt(pretraining(i,:), 1, 'db1');
    new_predata_swa(i,:) = swa;
    new_predata_swd(i,:) = swd;
end

for j = 1:tm
    [tswa, tswd] = swt(training(j,:), 1, 'db1');
    new_trdata_swa(j,:) = tswa;
    new_trdata_swd(j,:) = tswd;
end

save('DBN_data\data_10sec_dbn_ECGII_pretraining_swa.mat', 'new_predata_swa')
save('DBN_data\data_10sec_dbn_ECGII_pretraining_swd.mat', 'new_predata_swd')

save('DBN_data\data_10sec_dbn_ECGII_training_swa.mat', 'new_trdata_swa')
save('DBN_data\data_10sec_dbn_ECGII_training_swd.mat', 'new_trdata_swd')

csvwrite('DBN_data\data_10sec_dbn_ECGII_pretraining_swa.csv', new_predata_swa)
csvwrite('DBN_data\data_10sec_dbn_ECGII_pretraining_swd.csv', new_predata_swd)

csvwrite('DBN_data\data_10sec_dbn_ECGII_training_swa.csv', new_trdata_swa)
csvwrite('DBN_data\data_10sec_dbn_ECGII_training_swd.csv', new_trdata_swd)
