clc;
clear;

filelist = {'chal600', 'mimic600', 'chal300+mimic300', 'chal600+mimic600', 'chal600+mimic1200', 'chal600+mimic2400', 'chal600+mimicAll'};

length = 40000;

X = 1:length;
Y_accu = zeros(7,length);
Y_err = zeros(7,length);

for i = 1:7
    filename = filelist{i};
    test_accu = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_02/03_data/' filename '.h5'],'/test_accu');
    test_accu = test_accu(1:length);
    test_err = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_02/03_data/' filename '.h5'],'/test_err');
    test_err = test_err(1:length);
    Y_accu(i,:) = test_accu;
    Y_err(i,:) = test_err;
end

x_idx = 1:200:length;

s_Y_accu = zeros(7,200);
s_Y_err = zeros(7,200);
for j = 1:200
    start_idx = 200*(j-1)+1;
    end_idx = start_idx + 199;
    aver_Y_accu = mean(Y_accu(:,start_idx:end_idx), 2);
    aver_Y_err = mean(Y_err(:,start_idx:end_idx), 2);
    s_Y_accu(:,j) = aver_Y_accu;
    s_Y_err(:,j) = aver_Y_err;
end

figure
plot(x_idx,s_Y_err(1,:),'-o', ...
    x_idx,s_Y_err(2,:),'-+', ...
    x_idx,s_Y_err(3,:),'-*', ...
    x_idx,s_Y_err(4,:),'-.', ...
    x_idx,s_Y_err(5,:),'-x', ...
    x_idx,s_Y_err(6,:),'-s', ...
    x_idx,s_Y_err(7,:),'-d');
legend({'chal600', 'mimic600', 'chal300+mimic300', 'chal600+mimic600', 'chal600+mimic1200', 'chal600+mimic2400', 'chal600+mimicAll'}, 'Location','southeast', 'FontSize', 12)
xlabel('Update iteration', 'FontSize', 12)
ylabel('Negative log likelihood', 'FontSize', 12)
xlim([0 20000])
ylim([0.25 1.25])

figure
plot(x_idx,s_Y_accu(1,:),'-o', ...
    x_idx,s_Y_accu(2,:),'-+', ...
    x_idx,s_Y_accu(3,:),'-*', ...
    x_idx,s_Y_accu(4,:),'-.', ...
    x_idx,s_Y_accu(5,:),'-x', ...
    x_idx,s_Y_accu(6,:),'-s', ...
    x_idx,s_Y_accu(7,:),'-d');
legend({'chal600', 'mimic600', 'chal300+mimic300', 'chal600+mimic600', 'chal600+mimic1200', 'chal600+mimic2400', 'chal600+mimicAll'}, 'Location','southeast', 'FontSize', 12)
xlabel('Update iteration', 'FontSize', 12)
ylabel('Accuracy', 'FontSize', 12)
ylim([0.65 0.85])
