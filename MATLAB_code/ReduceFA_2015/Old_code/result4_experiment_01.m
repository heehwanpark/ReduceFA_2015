clc;
clear;

seed_list = {1, 2, 5, 10, 25};
type_list = {'chal', 'mimic+chal_all', 'mimic+chal_small', 'mimic2_all', 'mimic2_small'};
window_list = {50, 100, 150, 200, 250, 300}; 

X = 1:300;
Y_accu = zeros(6,300);
Y_err = zeros(6,300);

for i = 1:6
  data_type = type_list{2};
  db_seed = seed_list{1};
  init_seed = seed_list{1};
  window_size = window_list{i};
  
  filename = [data_type '_db_' int2str(db_seed) '_init_' int2str(init_seed) '_mw_' int2str(window_size) '.h5'];
  test_accu = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/05_max/' filename],'/test_accu');
  test_err = h5read(['/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_01/05_max/' filename],'/test_err');
  
  Y_accu(i,:) = test_accu;
  Y_err(i,:) = test_err;
end

figure
plot(X,Y_accu(1,:), X,Y_accu(2,:), X,Y_accu(3,:), X,Y_accu(4,:), X,Y_accu(5,:), X,Y_accu(6,:))
% legend({'chal(600)', 'mimic+chal(6348)', 'mimic+chal(600)', 'mimic2(5748)', 'mimic2(600)'},'Location','northwest', 'FontSize', 12)
legend({'window size: 50', 'window size: 100', 'window size: 150', 'window size: 200', 'window size: 250', 'window size: 300'},'Location','northwest', 'FontSize', 12)
% legend('1', '2', '5', '10', '25')

figure
plot(X,Y_err(1,:), X,Y_err(2,:), X,Y_err(3,:), X,Y_err(4,:), X,Y_err(5,:), X,Y_err(6,:))
% legend({'chal(600)', 'mimic+chal(6348)', 'mimic+chal(600)', 'mimic2(5748)', 'mimic2(600)'},'Location','northwest', 'FontSize', 12)
legend({'window size: 50', 'window size: 100', 'window size: 150', 'window size: 200', 'window size: 250', 'window size: 300'},'Location','northwest', 'FontSize', 12)
% legend('1', '2', '5', '10', '25')
ylim([0 1.5])
