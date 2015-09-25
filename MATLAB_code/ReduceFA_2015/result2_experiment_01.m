clc;
clear;

X = 1:300;
Y_accu = zeros(2,300);
Y_err = zeros(2,300);

folderlist = {'01_data/','04_wavelet/', '05_max/'};

for i = 1:3
  data_type = 'mimic+chal_all';
  db_seed = 1;
  init_seed = 1;
  foldername = folderlist{i};
  filename = [data_type '_db_' int2str(db_seed) '_init_' int2str(init_seed) '.h5'];
  test_accu = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/' foldername filename],'/test_accu');
  test_err = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/' foldername filename],'/test_err');
  
  Y_accu(i,:) = test_accu;
  Y_err(i,:) = test_err;
end

figure
plot(X,Y_accu(1,:),'-', X,Y_accu(2,:),'--', X,Y_accu(3,:),'-.')
legend('01','04', '05')

figure
plot(X,Y_err(1,:),'-', X,Y_err(2,:),'--', X,Y_err(3,:),'-.')
legend('01','04', '05')