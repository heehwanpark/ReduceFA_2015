input_data = h5read('D:\WFDB_data\MIT_BIH\dataset.h5', '/inputs');
target_data = h5read('D:\WFDB_data\MIT_BIH\dataset.h5', '/targets');

input_data = input_data';
target_data = target_data';

[im, in] = size(input_data);

new_input_swa = zeros(im,in);
new_input_swd = zeros(im,in);

for i = 1:im
    [swa, swd] = swt(input_data(i,:), 1, 'db1');
    new_input_swa(i,:) = swa;
    new_input_swd(i,:) = swd;
end

