cnn_errs = zeros(10,1);
cnn_accus = zeros(10,1);
martis_accus = zeros(10,1);

for i = 1:10
    file = strcat('0826_CNN_output\0826_result_10f_mitdb_',int2str(i),'_wo_pre.h5');
    test_err = h5read(file,'/test_err');
    test_accu = h5read(file,'/test_accu');
    min_err = find(test_err == min(test_err));
    cnn_accus(i) = test_accu(36);
end

for j = 1:10
    cmatrix = results_seg{j};
    accu = (cmatrix(1,1) + cmatrix(2,2))/sum(sum(cmatrix));
    martis_accus(j) = accu;
end

X = [cnn_accus martis_accus];
boxplot(X, 'labels', {'CNN','Martis'});