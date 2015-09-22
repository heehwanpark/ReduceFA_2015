seed_list = {1, 2, 5, 10, 25};
type_list = {'chal', 'mimic+chal_all', 'mimic+chal_small'};

tecol = zeros(5,300);
for i = 1:5
    filename = ['/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/mimic+chal_all_init_' int2str(seed_list{i}) '.h5'];
    test_err = h5read(filename, '/test_err');
    tecol(i,:) = test_err;
end

X = 1:300;
plot(X,tecol(1,:), X,tecol(2,:), X,tecol(3,:), X,tecol(4,:), X,tecol(5,:))
legend('1','2','5','10','25')