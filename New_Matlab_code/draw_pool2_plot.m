% MAX - MIN

clc;
clear;

mlp_list = {'[250-250]', '[500-500]', '[750-750]', '[1000-1000]', '[250-250-250]', '[500-500-500]', '[750-750-750]'};
feature_type = 'only_pool_x2';
max_iter = 200;

X = 1:max_iter;
Y_accu = zeros(length(mlp_list),max_iter);
Y_err = zeros(length(mlp_list),max_iter);

for j = 1:length(mlp_list)
    mlp_type = mlp_list{j};
    accu = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' mlp_type '-' feature_type '.h5'], '/test_accu');
    err = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' mlp_type '-' feature_type '.h5'], '/test_err');
    Y_accu(j,:) = accu(1:max_iter);
    Y_err(j,:) = err(1:max_iter);
end

a_X = 10:10:max_iter;
a_Y_accu = zeros(length(mlp_list),length(a_X));
a_Y_err = zeros(length(mlp_list),length(a_X));
for j = 1:length(a_X)
    start_idx = 10*(j-1)+1;
    end_idx = start_idx + 9;
    aver_Y_accu = mean(Y_accu(:,start_idx:end_idx), 2);
    aver_Y_err = mean(Y_err(:,start_idx:end_idx), 2);
    a_Y_accu(:,j) = aver_Y_accu;
    a_Y_err(:,j) = aver_Y_err;
end

figure
plot(a_X,a_Y_err(1,:),'-o', ...
    a_X,a_Y_err(2,:),'-+', ...
    a_X,a_Y_err(3,:),'-*', ...
    a_X,a_Y_err(4,:),'-.', ...
    a_X,a_Y_err(5,:),'-x', ...
    a_X,a_Y_err(6,:),'-s', ...
    a_X,a_Y_err(7,:),'-d');
legend(mlp_list, 'Location','northwest')
title('NLL of diffrent MLP structures with Max-pooling(4) layer')
xlabel('Epoch')
ylabel('Negative log likelihood')

figure
plot(a_X,a_Y_accu(1,:),'-o', ...
    a_X,a_Y_accu(2,:),'-+', ...
    a_X,a_Y_accu(3,:),'-*', ...
    a_X,a_Y_accu(4,:),'-.', ...
    a_X,a_Y_accu(5,:),'-x', ...
    a_X,a_Y_accu(6,:),'-s', ...
    a_X,a_Y_accu(7,:),'-d');
legend(mlp_list, 'Location','northwest')
title('Accuracy of diffrent MLP structures with Max-pooling(4) layer')
xlabel('Epoch')
ylabel('Accuracy')
