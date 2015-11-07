clc;
clear;

init_weight = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1106/KernelChange_init.h5', '/init_weight');
init_weight = init_weight';
inputs = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1106/KernelChange_init.h5', '/inputs');
inputs = inputs';
outputs_init = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1106/KernelChange_init.h5', '/outputs_init');
outputs_init = permute(outputs_init,[3 2 1]);

input = inputs(2,:);
output_2_init = squeeze(outputs_init(2,:,:));

new_input = input/10;
new_kernel = zeros(5,250);
for i = 1:5
    s_i = 250*(i-1)+1;
    e_i = s_i+250-1;
    new_kernel(i,:) = new_input(s_i:e_i);
end
new_weight = init_weight;
new_weight(1:5,:) = new_kernel;

h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1106/KernelChange_new.h5', '/new_weight', size(new_weight));
h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1106/KernelChange_new.h5', '/new_weight', new_weight);

%%

outputs_new = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1106/KernelChange_new.h5', '/outputs_new');
outputs_new = permute(outputs_new,[3 2 1]);

input = inputs(2,:);
output_2_new = squeeze(outputs_new(2,:,:));

%%
figure
plot(input)

figure
for i=1:15
    subplot(8,2,i)
    plot(new_weight(i,:))
end

figure
for i=1:15
    subplot(8,2,i)
    plot(output_2_init(i,:))
end

figure
for i=1:15
    subplot(8,2,i)
    plot(output_2_new(i,:))
end