clc;
clear;

chal_inputs = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_v3.h5','/inputs');
% chal_inputs = chal_inputs';
chal_targets = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_v3.h5','/targets');
% chal_targets = chal_targets';

[m,n] = size(chal_inputs);
window = 50;
x = 1:window;
y = normpdf(x,window/2,5);
nn = n-window+1;
maxmin_inputs = zeros(m,nn); 
gauss_inputs = zeros(m,nn);
for i=1:m
    ecg = chal_inputs(i,:);
    for j=1:nn
        s = j;
        e = s+window-1;
        maxmin_inputs(i,j) = max(ecg(s:e))-min(ecg(s:e));
        gauss_inputs(i,j) = sum(ecg(s:e).*y);
    end
end

%%
X = 1:nn;

figure
ax1_1 = subplot(3,1,1);
ax1_2 = subplot(3,1,2);
ax1_3 = subplot(3,1,3);

figure
ax2_1 = subplot(3,1,1);
ax2_2 = subplot(3,1,2);
ax2_3 = subplot(3,1,3);

plot(ax1_1, 1:2500, chal_inputs(1,:));
title(ax1_1, 'Original ECG');
plot(ax1_2, X, maxmin_inputs(1,:));
title(ax1_2, 'After MAX-MIN layer');
plot(ax1_3, X, gauss_inputs(1,:));
title(ax1_3, 'After Gaussian layer');

plot(ax2_1, 1:2500, chal_inputs(9,:));
title(ax2_1, 'Original ECG');
plot(ax2_2, X, maxmin_inputs(9,:));
title(ax2_2, 'After MAX-MIN layer');
plot(ax2_3, X, gauss_inputs(9,:));
title(ax2_3, 'After Gaussian layer');

% h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_gauss.h5','/inputs',size(new_inputs));
% h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_gauss.h5','/inputs',new_inputs);
% 
% h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_gauss.h5','/targets',size(chal_targets));
% h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_gauss.h5','/targets',chal_targets);