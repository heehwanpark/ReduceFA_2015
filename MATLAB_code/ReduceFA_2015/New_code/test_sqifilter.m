clc;
clear;

chal_input = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5', '/inputs');
mimic2_input = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_v3.h5', '/inputs');

% chal_input = chal_input';
% mimic2_input = mimic2_input';

testindex_chal = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1105/[500]-conv_test_result.h5', '/testindex_chal');
testindex_mimic = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1105/[500]-conv_test_result.h5', '/testindex_mimic');
pred_list = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1105/[500]-conv_test_result.h5', '/pred_list');
target_list = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1105/[500]-conv_test_result.h5', '/target_list');

chal_test = chal_input(testindex_chal, :);
mimic2_test = mimic2_input(testindex_mimic, :);
testset = [chal_test; mimic2_test];

mod_pred_list = pred_list;

TN_klist = [];
TN_slist = [];
FN_klist = [];
FN_slist = [];

for i = 1:256
    if pred_list(i) == 1
        neg_ecg = testset(i,:);
        k = kurtosis(neg_ecg);
        s = skewness(neg_ecg);
        if target_list(i) == 1
            TN_klist = [TN_klist k];
            TN_slist = [TN_slist s];
        else
            FN_klist = [FN_klist k];
            FN_slist = [FN_slist s];
        end 
    end
end

nonan_tnk = TN_klist(~isnan(TN_klist));
nonan_tns = TN_slist(~isnan(TN_slist));

nonan_fnk = FN_klist(~isnan(FN_klist));
nonan_fns = FN_slist(~isnan(FN_slist));

figure
histogram(nonan_tnk, 50)
hold on
histogram(nonan_fnk, 50)
title('kSQI(kurtosis)')
legend('True Negative','False Negative')

figure
histogram(nonan_tns, 50)
hold on
histogram(nonan_fns, 50)
title('sSQI(Skewness)')
legend('True Negative','False Negative')

% negative_set = testset(pred_list == 1);
% [m, ~] = size(negative_set);
% 
% k_list = zeros(1,m);
% s_list = zeros(1,m);
% for i = 1:m
%     neg_ecg = negative_set(i,:);
%     k = kurtosis(neg_ecg);
%     s = skewness(neg_ecg);
% end