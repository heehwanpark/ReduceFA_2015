clc;
clear;

chal_inputs = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5','/inputs');
chal_targets = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5','/targets');

mimic_inputs = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_v3.h5','/inputs');
mimic_targets = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_v3.h5','/targets');

nC = length(chal_targets);
nM = length(mimic_targets);

chal_inputs_wave = zeros(nC,4,2500);
mimic_inputs_wave = zeros(nM,4,2500);

for i = 1:nC
    input = chal_inputs(i,:);
    for j = 1:4
        c = ufwt(input,'db8',j);
        c = c';
        chal_inputs_wave(i,j,:) = c(1,:);
    end
end

for i = 1:nM
    input = mimic_inputs(i,:);
    for j = 1:4
        c = ufwt(input,'db8',j);
        c = c';
        mimic_inputs_wave(i,j,:) = c(1,:);
    end
end

origin = chal_inputs(500,:);
wavelets = chal_inputs_wave(500,:,:);
wavelets = squeeze(wavelets);

figure
subplot(5,1,1)
plot(origin)
title('Original signal')

subplot(5,1,2)
plot(wavelets(1,:));
title('Level 1 approximation')

subplot(5,1,3)
plot(wavelets(2,:));
title('Level 2 approximation')

subplot(5,1,4)
plot(wavelets(3,:));
title('Level 3 approximation')

subplot(5,1,5)
plot(wavelets(4,:));
title('Level 4 approximation')


% h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5','/wave_inputs', size(chal_inputs_wave));
% h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5','/wave_inputs', chal_inputs_wave);
% 
% h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_v3.h5','/wave_inputs', size(mimic_inputs_wave));
% h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_v3.h5','/wave_inputs', mimic_inputs_wave);
