% draw diff weight plot

% CNN

clc;
clear;

feature_type = 'conv';
max_iter = 200;
n_list = 2;

file_address = {'/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1126/[250-250]-false-conv-seed-1.h5',...
                '/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1208/[250-250]-conv-seed-1.h5'};

X = 1:max_iter;
Y_accu = zeros(n_list,max_iter);
Y_err = zeros(n_list,max_iter);
Y_score = zeros(n_list,max_iter);

for i = 1:n_list
    accu = h5read(file_address{i}, '/test_accu');
    err = h5read(file_address{i}, '/test_err');
    conf = h5read(file_address{i}, '/test_confmatrix');
    score = ([1 0 0 1]*conf)./([1 1 5 1]*conf);
    
    Y_accu(i,:) = accu(1:max_iter);
    Y_err(i,:) = err(1:max_iter);
    Y_score(i,:) = score(1:max_iter);
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

x_plot = X;
y_err_plot = Y_err;
y_accu_plot = Y_accu;
y_score_plot = Y_score;

figure
plot(x_plot,y_err_plot(1,:),'-o', ...
     x_plot,y_err_plot(2,:),'-+');
legend({'Initial', 'Modified'}, 'Location','southeast')
title('Negative log likelihood')
xlabel('Epoch')
ylabel('Negative log likelihood')

figure
plot(x_plot,y_accu_plot(1,:),'-o', ...
     x_plot,y_accu_plot(2,:),'-+');
legend({'Initial', 'Modified'}, 'Location','southeast')
title('Accuracy')
xlabel('Epoch')
ylabel('Accuracy')

figure
plot(x_plot,y_score_plot(1,:),'-o', ...
     x_plot,y_score_plot(2,:),'-+');
legend({'Initial', 'Modified'}, 'Location','southeast')
title('Challenge score')
xlabel('Epoch')
ylabel('Score')
