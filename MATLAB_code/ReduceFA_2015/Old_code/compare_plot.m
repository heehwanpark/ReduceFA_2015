
X = 1:200;

tr_err_1 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_chal_only_wo_pre.h5','/train_err');
te_err_1 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_chal_only_wo_pre.h5','/test_err');

tr_accu_1 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_chal_only_wo_pre.h5','/train_accu');
te_accu_1 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_chal_only_wo_pre.h5','/test_accu');

test_conf_1 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_chal_only_wo_pre.h5','/test_confmatrix');


tr_err_2 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_both_wo_pre.h5','/train_err');
te_err_2 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_both_wo_pre.h5','/test_err');

tr_accu_2 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_both_wo_pre.h5','/train_accu');
te_accu_2 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_both_wo_pre.h5','/test_accu');

test_conf_2 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_both_wo_pre.h5','/test_confmatrix');


% tr_err_3 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_mlp_3layer_500_wo_pre.h5','/train_err');
% te_err_3 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_mlp_3layer_500_wo_pre.h5','/test_err');
% 
% tr_accu_3 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_mlp_3layer_500_wo_pre.h5','/train_accu');
% te_accu_3 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_mlp_3layer_500_wo_pre.h5','/test_accu');
% 
% test_conf_3 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/0915_mlp_3layer_500_wo_pre.h5','/test_confmatrix');


figure
plot(X,tr_err_1,'o-', X,tr_err_2,'+-', X,te_err_1,'o-', X,te_err_2,'+-');
legend('Chal2015 (Train)','Chal2015+MIMIC (Train)', 'Chal2015 (Test)', 'Chal2015+MIMIC (Test)');
title('Negative Log Likelihood');


