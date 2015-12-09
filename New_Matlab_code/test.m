% test

clc;
clear;

test_accu = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1203/[500]-false-conv-seed-1.h5', '/test_accu');
test_err = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1203/[500]-false-conv-seed-1.h5', '/test_err');
test_conf = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1203/[500]-false-conv-seed-1.h5', '/test_confmatrix');

[m,n] = size(test_conf);
test_score = zeros(1,n);
for i=1:n
    conf_mat = test_conf(:,i);
    score = (conf_mat(4)+conf_mat(1))/(conf_mat(4)+conf_mat(2)+5*conf_mat(3)+conf_mat(1));
    test_score(i) = score;
end

figure
plot(test_accu)

figure
plot(test_err)

figure
plot(test_score)