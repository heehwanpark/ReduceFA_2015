pair_list = hdf5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/faultfile.h5', '/list');
pair_list = pair_list';

fail_list = [];
for i = 1:128
    if pair_list(i,2) == 0
        fail_list = [fail_list pair_list(i,1)];
    end
end
fail_list = sort(fail_list);

input_data = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5','/inputs');
target_data = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5','/targets');

fail_target = target_data(fail_list);

figure
k = 1;
for j = 1:length(fail_list)
    if fail_target(j) == 1
        subplot(6,2,k)
        plot(input_data(fail_list(j),:))
        k = k+1;
    end
end

