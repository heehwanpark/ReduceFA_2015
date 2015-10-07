clc;
clear;

filelist = {'chal600', 'mimic600', 'chal300+mimic300', 'chal600+mimic600', 'chal600+mimic1200', 'chal600+mimic2400', 'chal600+mimicAll'};

X = 1:20000;
Y_accu = zeros(7,20000);
Y_err = zeros(7,20000);

for i = 1:7
    filename = filelist{i};
    test_accu = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_02/01_data/' filename '.h5'],'/test_accu');
    test_accu = test_accu(1:20000);
    test_err = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_02/01_data/' filename '.h5'],'/test_err');
    test_err = test_err(1:20000);
    Y_accu(i,:) = test_accu;
    Y_err(i,:) = test_err;
end

x_idx = 1:200:20000;
s_Y_accu = Y_accu(:,x_idx);
s_Y_err = Y_err(:,x_idx);

CI = get(gca,'colororder');

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
ylim([0 1.5])

figure
plot(x_idx,s_Y_accu(1,:),'-o', x_idx,s_Y_accu(2,:),'-+', x_idx,s_Y_accu(3,:),'-*');
legend({'chal600', 'mimic600', 'chal300+mimic300'}, 'Location','southeast', 'FontSize', 12)
xlabel('Update iteration', 'FontSize', 12)
ylabel('Accuracy', 'FontSize', 12)
ylim([0.5 0.9])

figure
plot(x_idx,s_Y_accu(4,:),'-.', x_idx,s_Y_accu(5,:),'-x', x_idx,s_Y_accu(6,:),'-s', x_idx,s_Y_accu(7,:),'-d');
legend({'chal600+mimic600', 'chal600+mimic1200', 'chal600+mimic2400', 'chal600+mimicAll'}, 'Location','southeast', 'FontSize', 12)
xlabel('Update iteration', 'FontSize', 12)
ylabel('Accuracy', 'FontSize', 12)
ylim([0.5 0.9])