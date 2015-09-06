% Replicate the work of R.J.Martis et al, "ECG beat classification using PCA,
% LDA, ICA, and Discrete Wavelet Transfor", 2013

clc;
clear;

disp('Load dataset ...');
if exist('whole_peaks.mat','file') == 2 && exist('targets.mat','file') == 2
    load('whole_peaks.mat');
    load('targets.mat');
else
    datafolder = 'C:\Users\heehwan\workspace\Data\MIT_BIH\';
    filelist = ['100m'; '101m'; '102m'; '103m'; '104m'; '105m'; '106m'; '107m'; '108m'; '109m';
                '111m'; '112m'; '113m'; '114m'; '115m'; '116m'; '117m'; '118m'; '119m'; '121m';
                '122m'; '123m'; '124m'; '200m'; '201m'; '202m'; '203m'; '205m'; '207m'; '208m';
                '209m'; '210m'; '212m'; '213m'; '214m'; '215m'; '217m'; '219m'; '220m'; '221m';
                '222m'; '223m'; '228m'; '230m'; '231m'; '232m'; '233m'; '234m'];

    disp('Start pre-processing ...');
    whole_peaks = zeros(120000, 107*2);
    targets = zeros(120000, 1);
    cursor = 1;
    for i = 1:48
        filename = filelist(i,:);
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

                whole_peaks(cursor, :) = cA4_cD4;
                targets(cursor) = target;
                cursor = cursor+1;
            end
        end
    end
    whole_peaks(cursor:end, :) = [];
    targets(cursor:end) = [];
    
    save('whole_peaks.mat', 'whole_peaks');
    save('targets.mat', 'targets');
end

cA4 = whole_peaks(:,1:107);
cD4 = whole_peaks(:,107+1: end);

% PCA
disp('Apply PCA ...');
[coeff_cA, score_cA, latent_cA] = pca(cA4);
[coeff_cD, score_cD, latent_cD] = pca(cD4);

save('coeff_cA.mat','coeff_cA');
save('coeff_cD.mat','coeff_cD');

features = [score_cA(:,1:6) score_cD(:,1:6)];

% Training Set, Testing Set, 10-fold
disp('Prepare Training and Testing set ...')
rng(1);
[N, ~] = size(features);
shuffle_idx = randperm(N);
nTesting = round(N/10);
Training_idx = shuffle_idx(1:N-nTesting);
Testing_idx = shuffle_idx(N-nTesting+1:end);
Training_input = features(Training_idx,:);
Training_target = targets(Training_idx,:);
Testing_input = features(Testing_idx,:);
Testing_target = targets(Testing_idx,:);

% SVM - RBF
if exist('SVMModel.mat','file') == 2
    disp('Load SVM model ...');
    load('SVMModel.mat');
else
    disp('Learn SVM model ...');
    SVMModel = fitcsvm(Training_input,Training_target,'KernelFunction','rbf','Standardize',true);
    
    % % Multiclass
    % SVMModels = cell(5,1);
    % classes = unique(targets);
    % for k = 1:numel(classes)
    %     indx = (Training_target == classes(k));
    %     SVMModels{k} = fitcsvm(Training_input,indx,'KernelFunction','rbf','Standardize',true);
    % end
    
    save('SVMModel.mat', 'SVMModel');
end

% Prediction
[predictions, ~] = predict(SVMModel, Testing_input);
TP = 0;
FP = 0;
FN = 0;
TN = 0;
for i = 1:nTesting
    if Testing_target(i) == 1 && predictions(i) == 1
        TP = TP + 1;
    elseif Testing_target(i) == 2 && predictions(i) == 1
        FP = FP + 1;
    elseif Testing_target(i) == 1 && predictions(i) == 2
        FN = FN + 1;
    elseif Testing_target(i) == 2 && predictions(i) == 2
        TN = TN + 1;
    else
        disp('Something Wrong!!!')
        break
    end
end
sensitivity = TP/(TP+FN);
specificity = TN/(FP+TN);
accuracy = (TP+TN)/(TP+FP+FN+TN);
       