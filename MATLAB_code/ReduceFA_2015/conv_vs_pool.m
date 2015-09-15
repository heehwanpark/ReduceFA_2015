clc;
clear;

condition = [1	1	1	1	1	1	1	1	1	1;
            1	1	1	1	1	1	0	0	0	0;
            1	1	1	1	0	0	0	0	0	0;
            1	1	1	1	0	0	0	0	0	0;
            1	1	1	0	0	0	0	0	0	0;
            1	1	1	0	0	0	0	0	0	0;
            1	1	1	0	0	0	0	0	0	0;
            1	1	1	0	0	0	0	0	0	0;
            1	1	0	0	0	0	0	0	0	0;
            1	1	0	0	0	0	0	0	0	0];
       
condition = condition';

accu_mat = zeros(10,10);
score_mat = zeros(10,10);

folder = '/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/conv_vs_pool/';
for i = 1:10 % conv layer
    for j = 1:10 % pooling size
        if condition(i,j) == 1
            filename = ['conv_vs_pool_' int2str(i) '-' int2str(j) '_wo_pre.h5'];
            disp(filename)
            test_accu = h5read([folder filename], '/test_accu');
            test_err = h5read([folder filename], '/test_err');
            test_conf = h5read([folder filename], '/test_confmatrix');
            [~, min_i] = min(test_err);
            disp(test_accu(min_i))
            disp(max(test_accu))
            [~, max_i] = max(test_accu);
            accu_mat(i,j) = test_accu(min_i);
            min_conf = test_conf(:,max_i);
            TP = min_conf(4);
            TN = min_conf(1);
            FP = min_conf(2);
            FN = min_conf(3);
            score = (TP + TN)/(TP + TN + FP + 5*FN);
            disp(score)
            score_mat(i,j) = score;
        end
    end
end
            
        
