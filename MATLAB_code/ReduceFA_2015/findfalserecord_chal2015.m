pair_list = hdf5read('faultfile_chal.h5', '/list');
pair_list = pair_list';

fail_list = [];
for i = 1:145
    if pair_list(i,2) == 0
        fail_list = [fail_list pair_list(i,1)];
    end
end
fail_list = sort(fail_list);
