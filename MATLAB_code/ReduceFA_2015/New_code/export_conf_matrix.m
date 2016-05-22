% export Confusion matrix from each type

best_mlp = '[750-750-750]';
best_cnn = '[250-250]';
best_pool = '[500-500-500]';
best_poolx2 = '[750-750]';
best_maxmin = '[750-750]';
best_gauss = '[1000-1000]';

[mlp_best, mlp_idx] = max(h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' best_mlp '-mlp.h5'], '/test_accu'));
disp(mlp_best)
mlp_conf = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' best_mlp '-mlp.h5'], '/test_confmatrix');
best_mlp_conf = mlp_conf(:,mlp_idx);
disp([best_mlp_conf(4) best_mlp_conf(2); best_mlp_conf(3) best_mlp_conf(1)]);

[cnn_best, cnn_idx] = max(h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' best_cnn '-conv.h5'], '/test_accu'));
disp(cnn_best)
cnn_conf = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' best_cnn '-conv.h5'], '/test_confmatrix');
best_cnn_conf = cnn_conf(:,cnn_idx);
disp([best_cnn_conf(4) best_cnn_conf(2); best_cnn_conf(3) best_cnn_conf(1)]);

[pool_best, pool_idx] = max(h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' best_pool '-only_pool.h5'], '/test_accu'));
disp(pool_best)
pool_conf = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' best_pool '-only_pool.h5'], '/test_confmatrix');
best_pool_conf = pool_conf(:,pool_idx);
disp([best_pool_conf(4) best_pool_conf(2); best_pool_conf(3) best_pool_conf(1)]);

[poolx2_best, poolx2_idx] = max(h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' best_poolx2 '-only_pool_x2.h5'], '/test_accu'));
disp(poolx2_best)
poolx2_conf = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' best_poolx2 '-only_pool_x2.h5'], '/test_confmatrix');
best_poolx2_conf = poolx2_conf(:,poolx2_idx);
disp([best_poolx2_conf(4) best_poolx2_conf(2); best_poolx2_conf(3) best_poolx2_conf(1)]);

[maxmin_best, maxmin_idx] = max(h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' best_maxmin '-max-min.h5'], '/test_accu'));
disp(maxmin_best)
maxmin_conf = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' best_maxmin '-max-min.h5'], '/test_confmatrix');
best_maxmin_conf = maxmin_conf(:,maxmin_idx);
disp([best_maxmin_conf(4) best_maxmin_conf(2); best_maxmin_conf(3) best_maxmin_conf(1)]);

[gauss_best, gauss_idx] = max(h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1117/' best_gauss '-gauss.h5'], '/test_accu'));
disp(gauss_best)
gauss_conf = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1117/' best_gauss '-gauss.h5'], '/test_confmatrix');
best_gauss_conf = gauss_conf(:,gauss_idx);
disp([best_gauss_conf(4) best_gauss_conf(2); best_gauss_conf(3) best_gauss_conf(1)]);
