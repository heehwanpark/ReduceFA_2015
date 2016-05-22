clc;
clear;

X = (1:200)';

% test_err_4 = h5read('0806_c4_p2_chal-2015_wo_pre.h5','/test_err');
% test_err_5 = h5read('0806_c5_p2_chal-2015_wo_pre.h5','/test_err');
% test_err_6 = h5read('0806_c6_p2_chal-2015_wo_pre.h5','/test_err');
% test_err_7 = h5read('0806_c7_p2_chal-2015_wo_pre.h5','/test_err');
% 
% figure
% plot(X, test_err_4, 'o-', X, test_err_5, '+-', X, test_err_6, '*-', X, test_err_7, '.-');
% legend('4 conv error', '5 conv error', '6 conv error', '7 conv error');
% title('NLL for each conv layer');

% test_accu_4 = h5read('0806_c4_p2_chal-2015_wo_pre.h5','/test_accu');
% test_accu_5 = h5read('0806_c5_p2_chal-2015_wo_pre.h5','/test_accu');
% test_accu_6 = h5read('0806_c6_p2_chal-2015_wo_pre.h5','/test_accu');
% test_accu_7 = h5read('0806_c7_p2_chal-2015_wo_pre.h5','/test_accu');
% 
% figure
% plot(X, test_accu_4, 'o-', X, test_accu_5, '+-', X, test_accu_6, '*-', X, test_accu_7, '.-');
% legend('4 conv accu', '5 conv accu', '6 conv accu', '7 conv accu');
% title('Accuracy for each conv layer');

% test_err_42 = h5read('0806_c4_p2_chal-2015_wo_pre.h5','/test_err');
% test_err_42 = test_err_42(1:100);
% test_err_43 = h5read('0806_c4_p3_chal-2015_wo_pre.h5','/test_err');
% test_err_44 = h5read('0806_c4_p4_chal-2015_wo_pre.h5','/test_err');
% 
% figure
% plot(X, test_err_42, 'o-', X, test_err_43, '+-', X, test_err_44, '*-');
% legend('poolsize 2', 'poolsize 3', 'poolsize 4');
% title('NLL for each pool size');

% test_accu_42 = h5read('0806_c4_p2_chal-2015_wo_pre.h5','/test_accu');
% test_accu_42 = test_accu_42(1:100);
% test_accu_43 = h5read('0806_c4_p3_chal-2015_wo_pre.h5','/test_accu');
% test_accu_44 = h5read('0806_c4_p4_chal-2015_wo_pre.h5','/test_accu');
% 
% figure
% plot(X, test_accu_42, 'o-', X, test_accu_43, '+-', X, test_accu_44, '*-');
% legend('poolsize 2', 'poolsize 3', 'poolsize 4');
% title('Accuracy for each pool size');

% 
% test_err_44 = h5read('0806_c4_p4_chal-2015_wo_pre.h5','/test_err');
% test_err_53 = h5read('0806_c5_p3_chal-2015_wo_pre.h5','/test_err');
% test_err_62 = h5read('0806_c6_p2_chal-2015_wo_pre.h5','/test_err');
%  
% figure
% plot(X, test_err_44, 'o-', X, test_err_53, '+-', X, test_err_62, '*-');
% legend('C4 P4', 'C5 P3', 'C6 P2');
% title('NLL');


tr_err_1 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/0915_both_lr_wo_pre.h5','/train_err');
te_err_1 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/0915_both_lr_wo_pre.h5','/test_err');

tr_accu_1 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/0915_both_lr_wo_pre.h5','/train_accu');
te_accu_1 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/0915_both_lr_wo_pre.h5','/test_accu');

test_conf = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/0915_both_lr_wo_pre.h5','/test_confmatrix');

% tr_err_2 = h5read('0806_chal_result_2_wo_pre.h5','/train_err');
% te_err_2 = h5read('0806_chal_result_2_wo_pre.h5','/test_err');
% 
% tr_accu_2 = h5read('0806_chal_result_2_wo_pre.h5','/train_accu');
% te_accu_2 = h5read('0806_chal_result_2_wo_pre.h5','/test_accu');

figure
plot(X,tr_err_1,'o-', X,te_err_1,'+-');
legend('training','test');
title('Negative log likelihood');

figure
plot(X,tr_accu_1,'o-', X,te_accu_1,'+-');
legend('training','test');
title('Accuracy');