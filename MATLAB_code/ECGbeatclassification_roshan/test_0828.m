% Training Martis model with whole mitdb
% Test with challenge 2015

clc;
clear;

if exist('SVMModel_mitall.mat', 'file') == 2
    load('SVMModel_mitall.mat');
    load('coeff_cA_mitall.mat');
    load('coeff_cD_mitall.mat');
else
    mitdb_folder = 'C:\Users\heehwan\workspace\Data\Standard\mitdb\';
    mitdb_filelist = ['100m'; '101m'; '103m'; '105m'; '106m'; '107m'; '108m'; '109m';
        '111m'; '112m'; '113m'; '114m'; '115m'; '116m'; '117m'; '118m'; '119m'; '121m';
        '122m'; '123m'; '124m'; '200m'; '201m'; '202m'; '203m'; '205m'; '207m'; '208m';
        '209m'; '210m'; '212m'; '213m'; '214m'; '215m'; '217m'; '219m'; '220m'; '221m';
        '222m'; '223m'; '228m'; '230m'; '231m'; '232m'; '233m'; '234m'];

    rng(1);

    [SVMModel, coeff_cA, coeff_cD] = training_rjmartis_seperated(mitdb_folder, mitdb_filelist);

    save('SVMModel_mitall.mat', 'SVMModel');
    save('coeff_cA_mitall.mat', 'coeff_cA');
    save('coeff_cD_mitall.mat', 'coeff_cD');
end

% load challenge 2015 data
chal_input = h5read('C:\Users\heehwan\workspace\Data\chal2015_data_10sec_resampled.h5', '/input');
chal_target = h5read('C:\Users\heehwan\workspace\Data\chal2015_data_10sec_resampled.h5', '/target');

TP = 0;
FP = 0;
FN = 0;
TN = 0;

[N, ~] = size(chal_input);
preds = zeros(N,1);

for i = 1:N
%     if i == 11
%         disp('Find it!')
%     end
    ECG = chal_input(i,:);
    
    % Denoise ECG using by db6 wavelet transform
    denoised = denoising_rjmartis(ECG);
    
    % Pan-Tompkin QRS detection
    [~, qrs_i_raw, ~] = pan_tompkin(denoised, 360, false);
    
    nPeaks = length(qrs_i_raw);   
    
    peaks = zeros(nPeaks, 107*2);
    % Get peaks
    index = 1;
    for j = 1:nPeaks
        Rpoint = qrs_i_raw(j);
        if Rpoint >= 100 && Rpoint <= (length(denoised)-100)
            start_idx = Rpoint-99;
            end_idx = Rpoint+100;
            QRSpeak = denoised(start_idx:end_idx);
            
            [c_dmey, l_dmey] = wavedec(QRSpeak, 4, 'dmey');
            cA4_cD4 = c_dmey(1:(l_dmey(1)+l_dmey(2)));
            peaks(index, :) = cA4_cD4;
            
            index = index + 1;
        end
    end
    peaks(index:end,:) = [];
    
    cA4 = peaks(:,1:107);
    cD4 = peaks(:,107+1: end);
    
    % PCA
    score_cA = cA4*coeff_cA;
    score_cD = cD4*coeff_cD;
    
    features = [score_cA(:,1:6) score_cD(:,1:6)];
    
    % SVM - RBF
    [predictions, ~] = predict(SVMModel, features);
    
    % -------------------------------------
    % Asystole: No QRS for at least 4 seconds
    % Extreme Bradycardia: Heart rate lower than 40 bpm for 5 consecutive beats
    % Extreme Tachycardia: Heart rate higher than 140 bpm for 17 consecutive beats
    max_rr = 0;
    for q = 1:nPeaks-1
        cur_R = qrs_i_raw(q);
        next_R = qrs_i_raw(q+1);
        rr_interval = next_R - cur_R;
        if rr_interval > max_rr
            max_rr = rr_interval;
        end
    end
    
    if nPeaks < 7 % Bradycardia
        pred = 1;
    elseif nPeaks > 24 % Tachycardia
        pred = 1;
    elseif max_rr >= 4*360 || qrs_i_raw(end) <= 6*360 % Asystole
        pred = 1;
    elseif sum(predictions) > 0
        pred = 1;
    else         
        pred = 0;
    end
    % -------------------------------------
%     
%     if sum(predictions) > 0
%         pred = 1;
%     else
%         pred = 0;
%     end
    % -------------------------------------
    
    preds(i) = pred;
    
    target = chal_target(i);
    if target == 1 && pred == 1
        TP = TP + 1;
    elseif target == 0 && pred == 1
        FP = FP + 1;
    elseif target == 1 && pred == 0
        FN = FN + 1;
    elseif target == 0 && pred == 0
        TN = TN + 1;
    else
        disp('Something Wrong!!!')
        break
    end
end
    
confusion_matrix = [TP, FP; FN, TN];
disp(confusion_matrix)