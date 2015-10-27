clc;
clear;

filelist = {'[500]-chal600+mimicAll-max.h5', '[500]-chal600+mimicAll-min.h5', '[500]-chal600+mimicAll-normal.h5', 'chal600+mimicAll.h5'};

len = 40000;

X = 1:len;
Y_accu = zeros(4,len);
Y_err = zeros(4,len);

for i = 1:4
    filename = filelist{i};
    test_accu = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/dnn_output/' filename],'/test_accu');
    test_accu = test_accu(1:len);
    test_err = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/dnn_output/' filename],'/test_err');
    test_err = test_err(1:len);
    Y_accu(i,:) = test_accu;
    Y_err(i,:) = test_err;
end

x_idx = 1:200:len;

s_Y_accu = zeros(4,length(x_idx));
s_Y_err = zeros(4,length(x_idx));
for j = 1:length(x_idx)
    start_idx = 200*(j-1)+1;
    end_idx = start_idx + 199;
    aver_Y_accu = mean(Y_accu(:,start_idx:end_idx), 2);
    aver_Y_err = mean(Y_err(:,start_idx:end_idx), 2);
    s_Y_accu(:,j) = aver_Y_accu;
    s_Y_err(:,j) = aver_Y_err;
end
% s_Y_accu = Y_accu(:,x_idx);
% s_Y_err = Y_err(:,x_idx);

figure
plot(x_idx,s_Y_err(1,:),'-o', ...
    x_idx,s_Y_err(2,:),'-+', ...
    x_idx,s_Y_err(3,:),'-*', ...
    x_idx,s_Y_err(4,:),'-.');
legend({'max', 'min', 'normal', 'cnn'}, 'Location','southeast', 'FontSize', 12)
xlabel('Update iteration', 'FontSize', 12)
ylabel('Negative log likelihood', 'FontSize', 12)
xlim([0 40000])
ylim([0.25 1.25])

% figure
% plot(x_idx,s_Y_accu(1,:),'-o', ...
%     x_idx,s_Y_accu(2,:),'-+', ...
%     x_idx,s_Y_accu(3,:),'-*');
% legend({'max', 'min', 'normal'}, 'Location','southeast', 'FontSize', 12)
% xlabel('Update iteration', 'FontSize', 12)
% ylabel('Accuracy', 'FontSize', 12)