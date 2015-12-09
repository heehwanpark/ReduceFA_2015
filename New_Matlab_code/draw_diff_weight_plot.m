% draw diff weight plot

% CNN

clc;
clear;

weight_list = {'[0.5-0.5]', '[0.4-0.6]', '[0.3-0.7]', '[0.2-0.8]', '[0.15-0.85]', '[0.1-0.9]'};
feature_type = 'mlp';
max_iter = 200;
n_list = length(weight_list);

X = 1:max_iter;
Y_accu = zeros(n_list,max_iter);
Y_err = zeros(n_list,max_iter);
Y_score = zeros(n_list,max_iter);

for j = 1:n_list
    mlp_type = '[750-750]';
    file_address = ['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1203/' mlp_type '-' feature_type '-seed-1-' weight_list{j} '.h5'];
    accu = h5read(file_address, '/test_accu');
    err = h5read(file_address, '/test_err');
    conf = h5read(file_address, '/test_confmatrix');
    score = ([1 0 0 1]*conf)./([1 1 5 1]*conf);
    
    Y_accu(j,:) = accu(1:max_iter);
    Y_err(j,:) = err(1:max_iter);
    Y_score(j,:) = score(1:max_iter);
end

a_X = 10:10:max_iter;
a_Y_accu = zeros(n_list,length(a_X));
a_Y_err = zeros(n_list,length(a_X));
a_Y_score = zeros(n_list,length(a_X));
for j = 1:length(a_X)
    start_idx = 10*(j-1)+1;
    end_idx = start_idx + 9;
    aver_Y_accu = mean(Y_accu(:,start_idx:end_idx), 2);
    aver_Y_err = mean(Y_err(:,start_idx:end_idx), 2);
    aver_Y_score = mean(Y_score(:,start_idx:end_idx), 2);
    
    a_Y_accu(:,j) = aver_Y_accu;
    a_Y_err(:,j) = aver_Y_err;
    a_Y_score(:,j) = aver_Y_score;    
end

figure
plot(a_X,a_Y_err(1,:),'-o', ...
    a_X,a_Y_err(2,:),'-+', ...
    a_X,a_Y_err(3,:),'-*', ...
    a_X,a_Y_err(4,:),'-.', ...
    a_X,a_Y_err(5,:),'-x', ...
    a_X,a_Y_err(6,:),'-s');
legend(weight_list, 'Location','northeast')
title(['NLL of diffrent class weights: ' feature_type])
xlabel('Epoch')
ylabel('Negative log likelihood')

figure
plot(a_X,a_Y_accu(1,:),'-o', ...
    a_X,a_Y_accu(2,:),'-+', ...
    a_X,a_Y_accu(3,:),'-*', ...
    a_X,a_Y_accu(4,:),'-.', ...
    a_X,a_Y_accu(5,:),'-x', ...
    a_X,a_Y_accu(6,:),'-s');
legend(weight_list, 'Location','southeast')
title(['Accuracy of diffrent class weights: ' feature_type])
xlabel('Epoch')
ylabel('Accuracy')

figure
plot(a_X,a_Y_score(1,:),'-o', ...
    a_X,a_Y_score(2,:),'-+', ...
    a_X,a_Y_score(3,:),'-*', ...
    a_X,a_Y_score(4,:),'-.', ...
    a_X,a_Y_score(5,:),'-x', ...
    a_X,a_Y_score(6,:),'-s');
legend(weight_list, 'Location','southeast')
title(['Challenge score of diffrent class weights: ' feature_type])
xlabel('Epoch')
ylabel('Score')
