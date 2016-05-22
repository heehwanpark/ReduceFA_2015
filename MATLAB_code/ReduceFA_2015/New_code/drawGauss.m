clc;
clear;

norm_err = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/[250-250]-mlp.h5', '/test_err');
norm_accu = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/[250-250]-mlp.h5', '/test_accu');

maxmin_err = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/[250-250]-max-min.h5', '/test_err');
maxmin_accu = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1104/[250-250]-max-min.h5', '/test_accu');

gauss_err = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1110/[250-250]-mlp.h5', '/test_err');
gauss_accu = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1110/[250-250]-mlp.h5', '/test_accu');

X = 1:200;
plot(X,norm_accu, X,maxmin_accu, X,gauss_accu)
legend({'None', 'Max-Min', 'Gaussian'}, 'Location','northeast', 'FontSize', 12)
xlabel('Epochs', 'FontSize', 12)
ylabel('Accuracy', 'FontSize', 12)
ylim([0.3 1])
