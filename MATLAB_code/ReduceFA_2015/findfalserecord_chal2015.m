pair_list = hdf5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/faultfile.h5', '/list');
pair_list = pair_list';

fail_list = [];
for i = 1:128
    if pair_list(i,2) == 0
        fail_list = [fail_list pair_list(i,1)];
    end
end
fail_list = sort(fail_list);

