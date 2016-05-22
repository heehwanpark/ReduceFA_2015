% 11/26
clc;
clear;

db_seed_list = [1,2,3,5,7,11,13,17,19,23];
m = length(db_seed_list);

maxmin_type = '[750-750]';
conv_type = '[250-250]';

maxmin_accu = zeros(m,200);
maxmin_err = zeros(m,200);
maxmin_best = zeros(m,1);
maxmin_conf = zeros(m,4);

conv_accu = zeros(m,200);
conv_err = zeros(m,200);
conv_best = zeros(m,1);
conv_conf = zeros(m,4);

for i = 1:m
    mm_accu = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1125/' maxmin_type '-max-min-seed-' int2str(db_seed_list(i)) '.h5'], '/test_accu');
    mm_err = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1125/' maxmin_type '-max-min-seed-' int2str(db_seed_list(i)) '.h5'], '/test_err');
    mm_conf = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1125/' maxmin_type '-max-min-seed-' int2str(db_seed_list(i)) '.h5'], '/test_confmatrix');
    
    maxmin_accu(i,:) = mm_accu;
    maxmin_err(i,:) = mm_err;
    
    [mm_best, mm_idx] = max(mm_accu);
    mm_bestconf = mm_conf(:,mm_idx);

    disp(['maxmin' db_seed_list(i)])
    disp(mm_best)
    disp(mm_idx)
    disp([mm_bestconf(4) mm_bestconf(2); mm_bestconf(3) mm_bestconf(1)])
    
    maxmin_best(i) = mm_best;
    maxmin_conf(i,:) = mm_bestconf';
    
    c_accu = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1125/' conv_type '-false-conv-seed-' int2str(db_seed_list(i)) '.h5'], '/test_accu');
    c_err = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1125/' conv_type '-false-conv-seed-' int2str(db_seed_list(i)) '.h5'], '/test_err');
    c_conf = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1125/' conv_type '-false-conv-seed-' int2str(db_seed_list(i)) '.h5'], '/test_confmatrix');
    
    conv_accu(i,:) = c_accu;
    conv_err(i,:) = c_err;
    
    [c_best, c_idx] = max(c_accu);
    c_bestconf = c_conf(:,c_idx);    
    
    disp(['conv' db_seed_list(i)])
    disp(c_best)
    disp(c_idx)
    disp([c_bestconf(4) c_bestconf(2); c_bestconf(3) c_bestconf(1)])
    
    conv_best(i) = c_best;
    conv_conf(i,:) = c_bestconf';    
end

boxplot([maxmin_best conv_best], {'Max-Min + DNN', 'CNN'})
title('Accuracy')
ylabel('Accuracy')

