clc;
clear;

% upsampled data - 360Hz
inputs = h5read('C:\Users\heehwan\workspace\Data\chal2015_data_10sec_resampled.h5', '/input');
targets = h5read('C:\Users\heehwan\workspace\Data\chal2015_data_10sec_resampled.h5', '/target');

load('SVMModel.mat');

load('coeff_cA.mat');
load('coeff_cD.mat');

TP = 0;
FP = 0;
FN = 0;
TN = 0;

N = length(targets);
for i = 1:N
    ECG = inputs(i,:);
    [C, L] = wavedec(ECG, 9, 'db6');
    remains = sum(L(1:8));
    nc = C;
    nc(remains+1:end) = 0;
    den_ECG = waverec(nc, L, 'db6');
    
    [qrs_amp_raw,qrs_i_raw,delay] = pan_tompkin(den_ECG, 360, false);
    cood = qrs_i_raw;
    
    peaks = zeros(length(cood),107+107);
    cursor = 1;
    for j = 1:length(cood)
        Rpeak = cood(j);
        if Rpeak >= 100 && (3600-Rpeak) >= 100
            start_idx = Rpeak-99;
            end_idx = Rpeak+100;
            QRSpeak = den_ECG(start_idx:end_idx);
            % DWT using dmey
            [c_dmey, l_dmey] = wavedec(QRSpeak, 4, 'dmey');
            cA4_cD4 = c_dmey(1:(l_dmey(1)+l_dmey(2)));
            
            peaks(cursor, :) = cA4_cD4;
            cursor = cursor+1;
        end
    end
    peaks(cursor:end, :) = [];
    
    cA4 = peaks(:,1:107);
    cD4 = peaks(:,107+1:end);
    
    % PCA
    score_cA = cA4*coeff_cA;
    score_cD = cD4*coeff_cD;
    
    features = [score_cA(:,1:6) score_cD(:,1:6)];
    
    [label_for_peak, ~] = predict(SVMModel, features);
    label_for_peak = label_for_peak - 1;
    
    target = targets(i);
    if sum(label_for_peak) > 0
        prediction = 1; % Abnormal
    else
        prediction = 0; % Normal
    end
    
    if target == 0 && prediction == 0
        TP = TP + 1;
    elseif target == 1 && prediction == 0
        FP = FP + 1;
    elseif target == 0 && prediction == 1
        FN = FN + 1;
    elseif target == 1 && prediction == 1
        TN = TN + 1;
    else
        disp('Something Wrong!!!');
        break
    end    
end
confusion_matrix = [TP FP; FN TN];
disp(confusion_matrix)

sensitivity = TP/(TP+FN);
specificity = TN/(FP+TN);
accuracy = (TP+TN)/(TP+FP+FN+TN);
disp(sensitivity)
disp(specificity)
disp(accuracy)