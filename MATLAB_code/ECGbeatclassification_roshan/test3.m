sens = zeros(1,10);
spec = zeros(1,10);
ppv = zeros(1,10);
accu = zeros(1,10);

for i = 1:10
    cmatrix = cell_confusion_matrix{i};
    TN = cmatrix(1,1);
    FN = cmatrix(1,2);
    FP = cmatrix(2,1);
    TP = cmatrix(2,2);
    sens(i) = TP/(TP+FN);
    spec(i) = TN/(TN+FP);
    ppv(i) = TP/(TP+FP);
    accu(i) = (TP+TN)/(TP+FP+FN+TN);
end

disp(mean(sens))
disp(mean(spec))
disp(mean(ppv))
disp(mean(accu))
