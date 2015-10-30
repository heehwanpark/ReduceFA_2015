clc;
clear;

mlp_list = {'[500]', '[1000]', '[500-500]', '[1000-1000]', '[500-500-500]'};
feature_list = {'conv', 'max', 'min', 'max-min'};

len = 40000;

for i = 2:4
    feature_type = feature_list{i};
    
    X = 1:len;
    Y_accu = zeros(5,len);
    Y_err = zeros(5,len);
    for j = 1:5
        mlp_type = mlp_list{j};
        accu = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/' mlp_type '-' feature_type], '/test_accu');
        err = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/' mlp_type '-' feature_type], '/test_err');
        Y_accu(j,:) = accu;
        Y_err(j,:) = err;
    end
    
    x_idx = 1:200:len;
    s_Y_accu = zeros(5,length(x_idx));
    s_Y_err = zeros(5,length(x_idx));
    for k = 1:length(x_idx)
        start_idx = 200*(k-1)+1;
        end_idx = start_idx + 199;
        aver_Y_accu = mean(Y_accu(:,start_idx:end_idx), 2);
        aver_Y_err = mean(Y_err(:,start_idx:end_idx), 2);
        s_Y_accu(:,k) = aver_Y_accu;
        s_Y_err(:,k) = aver_Y_err;
    end
    
    figure
    plot(x_idx,s_Y_err(1,:),'-o', ...
        x_idx,s_Y_err(2,:),'-+', ...
        x_idx,s_Y_err(3,:),'-*', ...
        x_idx,s_Y_err(4,:),'-.', ...
        x_idx,s_Y_err(5,:),'-d');
    legend({'[500]', '[1000]', '[500-500]', '[1000-1000]', '[500-500-500]'}, 'Location','southeast', 'FontSize', 12)
    xlabel('Update iteration', 'FontSize', 12)
    ylabel('Negative log likelihood', 'FontSize', 12)
    xlim([0 40000])
    ylim([0.25 1.25])
end