clc;
clear;

% 11/10
mlp_type = '[500]';
feature_list = {'true-conv','false-conv'};
f_name = {'Random init', 'Art init'};
max_iter = 200;

X = 1:max_iter;
Y_accu = zeros(length(feature_list),max_iter);
Y_err = zeros(length(feature_list),max_iter);

for i = 1:length(feature_list)
    feature_type = feature_list{i};
    accu = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1109/' mlp_type '-' feature_type '.h5'], '/test_accu');
    err = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1109/' mlp_type '-' feature_type '.h5'], '/test_err');
    Y_accu(i,:) = accu(1:max_iter);
    Y_err(i,:) = err(1:max_iter);
end
    
figure
plot(X,Y_err(1,:),'-o', ...
    X,Y_err(2,:),'-+');
legend(f_name, 'Location','northeast', 'FontSize', 12)
xlabel('Update iteration', 'FontSize', 12)
ylabel('Negative log likelihood', 'FontSize', 12)
ylim([0.3 1])

figure
plot(X,Y_accu(1,:),'-o', ...
    X,Y_accu(2,:),'-+');
legend(f_name, 'Location','northeast', 'FontSize', 12)
xlabel('Update iteration', 'FontSize', 12)
ylabel('Accuracy', 'FontSize', 12)
ylim([0.3 1])

%%

inputs = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1109/conv_through.h5', '/inputs');
inputs = inputs';

outputs1 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1109/conv_through.h5', '/outputs1');
outputs1 = permute(outputs1,[3 2 1]);

outputs2 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1109/conv_through.h5', '/outputs2');
outputs2 = permute(outputs2,[3 2 1]);

outputs3 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1109/conv_through.h5', '/outputs3');
outputs3 = permute(outputs3,[3 2 1]);

inputs_1 = inputs(2,:);
outputs1_1 = squeeze(outputs1(2,:,:));
outputs2_1 = squeeze(outputs2(2,:,:));
outputs3_1 = squeeze(outputs3(2,:,:));

figure
for i=1:10
    subplot(5,2,i)
    if i <= 5
        plot(outputs1_1(i,:))
    else
        plot(outputs1_1(i+50,:))
    end
end

figure
for i=1:10
    subplot(5,2,i)
    if i <= 5
        plot(outputs2_1(i,:))
    else
        plot(outputs2_1(i+50,:))
    end
end

figure
for i=1:10
    subplot(5,2,i)
    if i <= 5
        plot(outputs3_1(i,:))
    else
        plot(outputs3_1(i+50,:))
    end
end
