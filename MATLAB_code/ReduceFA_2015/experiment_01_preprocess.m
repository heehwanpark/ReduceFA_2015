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

h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5','/wave_inputs', size(chal_inputs_wave));
h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5','/wave_inputs', chal_inputs_wave);

h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_v3.h5','/wave_inputs', size(mimic_inputs_wave));
h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_v3.h5','/wave_inputs', mimic_inputs_wave);
