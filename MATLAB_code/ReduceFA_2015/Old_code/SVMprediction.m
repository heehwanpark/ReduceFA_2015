function [accuracy, sensitivity, specificity] = SVMprediction(SVMModel, testing_input, testing_target)
    [predictions, ~] = predict(SVMModel, testing_input);
    TP = 0;
    FP = 0;
    FN = 0;
    TN = 0;
    for i = 1:nTesting
        if testing_target(i) == 1 && predictions(i) == 1
            TP = TP + 1;
        elseif testing_target(i) == 2 && predictions(i) == 1
            FP = FP + 1;
        elseif testing_target(i) == 1 && predictions(i) == 2
            FN = FN + 1;
        elseif testing_target(i) == 2 && predictions(i) == 2
            TN = TN + 1;
        else
            disp('Something Wrong!!!')
            break
        end
    end
    sensitivity = TP/(TP+FN);
    specificity = TN/(FP+TN);
    accuracy = (TP+TN)/(TP+FP+FN+TN);       
end