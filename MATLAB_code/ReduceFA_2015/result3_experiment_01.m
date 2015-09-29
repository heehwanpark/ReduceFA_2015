clc;
clear;

seed_list = {1, 2, 5, 10, 25};
type_list = {'chal', 'mimic+chal_all', 'mimic+chal_small', 'mimic2_all', 'mimic2_small'};

X = 1:300;

for i = 1:5
  data_type = type_list{2};
  db_seed = seed_list{i};
  init_seed = seed_list{1};
  
  filename = [data_type '_db_' int2str(db_seed) '_init_' int2str(init_seed) '.h5'];
  err_03 = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/03_difftest/' filename],'/test_err');
  err_04 = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/04_wavelet/' filename],'/test_err');
  
  figure
  plot(X,err_03, X,err_04);
  title(['DB seed ' int2str(db_seed)])
  legend('vanilla', 'wavelet');
  ylim([0 1.5]);
end
