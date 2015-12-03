% test

clc;
clear;

[maxmin_best, maxmin_idx] = max(h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1125/[750-750]-max-min-seed-1.h5', '/test_accu'));
disp(maxmin_best)
maxmin_conf = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1125/[750-750]-max-min-seed-1.h5', '/test_confmatrix');
best_maxmin_conf = maxmin_conf(:,maxmin_idx);
disp([best_maxmin_conf(4) best_maxmin_conf(2); best_maxmin_conf(3) best_maxmin_conf(1)]);
