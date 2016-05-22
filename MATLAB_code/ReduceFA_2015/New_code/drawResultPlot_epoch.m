clc;
clear;

% 11/4
% mlp_list = {'[250-250]', '[500-500]', '[750-750]', '[1000-1000]', '[250-250-250]', '[500-500-500]', '[750-750-750]'};
% feature_list = {'conv', 'mmconv', 'mlp', 'max-min', 'max-min_x2', 'only_pool', 'only_pool_x2', 'mmpool'};
% f_name = {'Convolutional', '(MAX-MIN)-Conv', 'None', 'MAX-MIN', 'MAX-MIN x2', 'only pooling', 'only pooling x2', '(MAX-MIN)-pooling'};

% 11/10
mlp_list = '[500]';
feature_list = {'true-conv','false-conv'};
f_name = {'Random init', 'Art init'};
max_iter = 200;

% for i = 1:length(feature_list)
%     feature_type = feature_list{i};
%     
%     X = 1:max_iter;
%     Y_accu = zeros(length(mlp_list),max_iter);
%     Y_err = zeros(length(mlp_list),max_iter);
%     for j = 1:length(mlp_list)
%         mlp_type = mlp_list{j};
%         accu = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' mlp_type '-' feature_type '.h5'], '/test_accu');
%         err = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' mlp_type '-' feature_type '.h5'], '/test_err');
%         Y_accu(j,:) = accu(1:max_iter);
%         Y_err(j,:) = err(1:max_iter);
%     end
%     
%     figure
%     plot(X,Y_err(1,:),'-o', ...
%         X,Y_err(2,:),'-+', ...
%         X,Y_err(3,:),'-*', ...
%         X,Y_err(4,:),'-.', ...
%         X,Y_err(5,:),'-x', ...
%         X,Y_err(6,:),'-s', ...
%         X,Y_err(7,:),'-d');
%     legend({'[250-250]', '[500-500]', '[750-750]', '[1000-1000]', '[250-250-250]', '[500-500-500]', '[750-750-750]'}, 'Location','northeast', 'FontSize', 12)
%     title(f_name{i})
%     xlabel('Update iteration', 'FontSize', 12)
%     ylabel('Negative log likelihood', 'FontSize', 12)
%     ylim([0.3 1])
% end

X = 1:max_iter;
Y_accu = zeros(length(feature_list),max_iter);
Y_err = zeros(length(feature_list),max_iter);
mlp_type = mlp_list{1};

for i = 1:length(feature_list)
    feature_type = feature_list{i};
    accu = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' mlp_type '-' feature_type '.h5'], '/test_accu');
    err = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/' mlp_type '-' feature_type '.h5'], '/test_err');
    Y_accu(i,:) = accu(1:max_iter);
    Y_err(i,:) = err(1:max_iter);
end
    
figure
plot(X,Y_err(1,:),'-o', ...
    X,Y_err(2,:),'-+', ...
    X,Y_err(3,:),'-*', ...
    X,Y_err(4,:),'-.', ...
    X,Y_err(5,:),'-x', ...
    X,Y_err(6,:),'-s', ...
    X,Y_err(7,:),'-d', ...
    X,Y_err(8,:),'->');
legend(f_name, 'Location','northeast', 'FontSize', 12)
xlabel('Update iteration', 'FontSize', 12)
ylabel('Negative log likelihood', 'FontSize', 12)
ylim([0.3 1])

figure
plot(X,Y_accu(1,:),'-o', ...
    X,Y_accu(2,:),'-+', ...
    X,Y_accu(3,:),'-*', ...
    X,Y_accu(4,:),'-.', ...
    X,Y_accu(5,:),'-x', ...
    X,Y_accu(6,:),'-s', ...
    X,Y_accu(7,:),'-d', ...
    X,Y_accu(8,:),'->');
legend(f_name, 'Location','northeast', 'FontSize', 12)
xlabel('Update iteration', 'FontSize', 12)
ylabel('Accuracy', 'FontSize', 12)
ylim([0.3 1])
