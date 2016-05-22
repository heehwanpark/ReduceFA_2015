% clc;
% clear;
% 
% conv01_origin = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/extract_conv_layer_conv_origin.h5','/conv1_weight');
% conv02_origin = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/extract_conv_layer_conv_origin.h5','/conv2_weight');
% 
% conv01_weights = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/extract_conv_layer.h5','/conv1_weight');
% conv02_weights = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/extract_conv_layer.h5','/conv2_weight');
% 
% c1_o = conv01_origin(:,:)';
% c1_100 = conv01_weights(:,:,100)';
% c1_200 = conv01_weights(:,:,200)';
% 
% i = 50;
% X = 1:250;
% plot(X,c1_o(i,:),X,c1_100(i,:),X,c1_200(i,:))
% legend('origin','100','200')


% [m,n] = size(inputs);
% window = 100;
% x = 1:window;
% y = normpdf(x,window/2,5);
% nn = n-window+1;
% new_ecg = zeros(m,nn);
% for i=1:m
%     ecg = inputs(i,:);
%     for j=1:nn
%         s = j;
%         e = s+window-1;
%         new_ecg(i,j) = sum(ecg(s:e).*y);
% %         new_ecg(i,j) = max(ecg(s:e))-min(ecg(s:e));
%     end
% end
        