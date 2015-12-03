% draw kernel

clc;
clear;

init_weight_1 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1125/kernelFromModel.h5', '/init_weight_1');
init_weight_1 = init_weight_1';

train_weight_1 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1125/kernelFromModel.h5', '/train_weight_1');
train_weight_1 = train_weight_1';

X = 1:50;
plot(X,init_weight_1(50,:),'-o', ...
     X,train_weight_1(50,:), '-+');
legend({'Initial kernel','Trained kernel'});

% for i = 41:50
%     subplot(5,2,i-40)
%     plot(X,init_weight_1(i,:), X,train_weight_1(i,:))
%     legend({'Initial kernel','Trained kernel'});
% end