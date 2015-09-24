clc;
clear;

seed_list = {1, 2, 5, 10, 25};
type_list = {'chal', 'mimic+chal_all', 'mimic+chal_small', 'mimic2_all', 'mimic2_small'};

X = 1:300;
Y_accu = zeros(5,300);
Y_err = zeros(5,300);

for i = 1:5
  data_type = type_list{2};
  db_seed = seed_list{i};
  init_seed = seed_list{1};
  
  filename = [data_type '_db_' int2str(db_seed) '_init_' int2str(init_seed) '.h5'];
  test_accu = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/03_difftest/' filename],'/test_accu');
  test_err = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/03_difftest/' filename],'/test_err');
  
  Y_accu(i,:) = test_accu;
  Y_err(i,:) = test_err;
end

figure
plot(X,Y_accu(1,:),'-', X,Y_accu(2,:),'--', X,Y_accu(3,:),'-o', X,Y_accu(4,:),'-.', X,Y_accu(5,:),'-+')
legend('1', '2', '5', '10', '25')

figure
plot(X,Y_err(1,:),'-', X,Y_err(2,:),'--', X,Y_err(3,:),'-o', X,Y_err(4,:),'-.', X,Y_err(5,:),'-+')
legend('1', '2', '5', '10', '25')