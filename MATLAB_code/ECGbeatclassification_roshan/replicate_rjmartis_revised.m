% Replicate the work of R.J.Martis et al, "ECG beat classification using PCA,
% LDA, ICA, and Discrete Wavelet Transfor", 2013

clc;
clear;

disp('Load dataset ...');
datafolder = 'C:\Users\heehwan\workspace\Data\MIT_BIH\';
filelist = ['100m'; '101m'; '102m'; '103m'; '104m'; '105m'; '106m'; '107m'; '108m'; '109m';
    '111m'; '112m'; '113m'; '114m'; '115m'; '116m'; '117m'; '118m'; '119m'; '121m';
    '122m'; '123m'; '124m'; '200m'; '201m'; '202m'; '203m'; '205m'; '207m'; '208m';
    '209m'; '210m'; '212m'; '213m'; '214m'; '215m'; '217m'; '219m'; '220m'; '221m';
    '222m'; '223m'; '228m'; '230m'; '231m'; '232m'; '233m'; '234m'];

rng(1);
shuffle_idx = randperm(43);

trainings = filelist(shuffle_idx(1:43),:);
% testings = filelist(shuffle_idx(41:48),:);

trainings_peaks = zeros(110000, 107*2);
trainings_targets = zeros(110000, 1);
cursor = 1;
for i = 1:48
    filename = trainings(i,:);
    filenum = filename(1:3);
    disp(filename);
    
    ecgfile = load(strcat(datafolder, filename, '.mat'));
    dataset = ecgfile.val;
    ECG = dataset(1,:);
    
    % Denosing
    [C, L] = wavedec(ECG, 9, 'db6');
    remains = sum(L(1:8));
    nc = C;
    nc(remains+1:end) = 0;
    den_ECG2 = waverec(nc, L, 'db6');
    
    % Load heartbeat
    [cood, annot] = getAnnotation(filenum,datafolder);
    
    % Pan-Tompkin QRS detection
    %     [qrs_amp_raw,qrs_i_raw,delay] = pan_tompkin(den_ECG2, 360, false);
    %     cood = qrs_i_raw % No annotation now
    
    for j = 1:length(cood)
        Rpeak = cood(j);
        target = annot(j);
        if Rpeak >= 100 && (650000-Rpeak) >= 100
            start_idx = Rpeak-99;
            end_idx = Rpeak+100;
            QRSpeak = den_ECG2(start_idx:end_idx);
            % DWT using dmey
            [c_dmey, l_dmey] = wavedec(QRSpeak, 4, 'dmey');
            cA4_cD4 = c_dmey(1:(l_dmey(1)+l_dmey(2)));
            
            trainings_peaks(cursor, :) = cA4_cD4;
            trainings_targets(cursor) = target;
            cursor = cursor+1;
        end
    end
end
trainings_peaks(cursor:end, :) = [];
trainings_targets(cursor:end) = [];

cA4 = trainings_peaks(:,1:107);
cD4 = trainings_peaks(:,107+1: end);

% PCA
disp('Apply PCA ...');
[coeff_cA, score_cA, latent_cA] = pca(cA4, 'Centered', false);
[coeff_cD, score_cD, latent_cD] = pca(cD4, 'Centered', false );

save('coeff_cA.mat', 'coeff_cA');
save('coeff_cD.mat', 'coeff_cD');
    
features = [score_cA(:,1:6) score_cD(:,1:6)];

% SVM - RBF
disp('Learn SVM model ...');
SVMModel = fitcsvm(features, trainings_targets,'KernelFunction','rbf','Standardize',true);
save('SVMModel.mat', 'SVMModel');

%%
%%%%%%%%%%%
% Testing %
%%%%%%%%%%%
disp('Start testing ...')
testings_peaks = zeros(50000, 107*2);
testings_targets = zeros(50000, 1);
cursor = 1;
for i = 1:8
    filename = testings(i,:);
    filenum = filename(1:3);
    disp(filename);
    
    ecgfile = load(strcat(datafolder, filename, '.mat'));
    dataset = ecgfile.val;
    ECG = dataset(1,:);
    
    % Denosing
    [C, L] = wavedec(ECG, 9, 'db6');
    remains = sum(L(1:8));
    nc = C;
    nc(remains+1:end) = 0;
    den_ECG2 = waverec(nc, L, 'db6');
    
    % Load heartbeat
    [cood, annot] = getAnnotation(filenum,datafolder);
    
    % Pan-Tompkin QRS detection
    %     [qrs_amp_raw,qrs_i_raw,delay] = pan_tompkin(den_ECG2, 360, false);
    %     cood = qrs_i_raw % No annotation now
    
    for j = 1:length(cood)
        Rpeak = cood(j);
        target = annot(j);
        if Rpeak >= 100 && (650000-Rpeak) >= 100
            start_idx = Rpeak-99;
            end_idx = Rpeak+100;
            QRSpeak = den_ECG2(start_idx:end_idx);
            % DWT using dmey
            [c_dmey, l_dmey] = wavedec(QRSpeak, 4, 'dmey');
            cA4_cD4 = c_dmey(1:(l_dmey(1)+l_dmey(2)));
            
            testings_peaks(cursor, :) = cA4_cD4;
            testings_targets(cursor) = target;
            cursor = cursor+1;
        end
    end
end
testings_peaks(cursor:end, :) = [];
testings_targets(cursor:end) = [];

cA4_t = testings_peaks(:,1:107);
cD4_t = testings_peaks(:,107+1: end);

% PCA
[num_peaks, ~] = size(cA4_t);
score_cA_t = cA4_t*coeff_cA;
score_cD_t = cD4_t*coeff_cD;

features_t = [score_cA_t(:,1:6) score_cD_t(:,1:6)];
[predictions, ~] = predict(SVMModel, features_t);

TP = 0;
FP = 0;
FN = 0;
TN = 0;
for i = 1:num_peaks
    if testings_targets(i) == 1 && predictions(i) == 1
        TP = TP + 1;
    elseif testings_targets(i) == 2 && predictions(i) == 1
        FP = FP + 1;
    elseif testings_targets(i) == 1 && predictions(i) == 2
        FN = FN + 1;
    elseif testings_targets(i) == 2 && predictions(i) == 2
        TN = TN + 1;
    else
        disp('Something Wrong!!!')
        break
    end
end
sensitivity = TP/(TP+FN);
specificity = TN/(FP+TN);
accuracy = (TP+TN)/(TP+FP+FN+TN);

disp(sensitivity)
disp(specificity)
disp(accuracy)