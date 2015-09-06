function confusion_matrix = test_rjmartis_last_chal2015(SVMModel, coeff_cA, coeff_cD, datafolder, testlist)
[N, ~] = size(testlist);
num_segments = floor(650000/3600);
targets = zeros(N*num_segments, 1);
predictions = zeros(N*num_segments, 1);

idx = 1;
for i = 1:N
    filename = testlist(i,:);
    disp(strcat('... ', filename))
    datafile = load(strcat(datafolder, filename, '.mat'));
    dataset = datafile.standard_dataset;
    ECG = dataset(1,:);
    labels = dataset(2,:);
    
    for j = 1:num_segments
        start_idx = (j-1)*3600+1;
        end_idx = start_idx+3600-1;
        segment_ecg = ECG(start_idx:end_idx);
        segment_labels = labels(start_idx:end_idx);
        
        % Denoise ECG using by db6 wavelet transform
        denoised = denoising_rjmartis(segment_ecg);
        % Pan-Tompkin QRS detection
        [~, qrs_i_raw, ~] = pan_tompkin(denoised, 360, false);
        
        % Get peaks
        [peaks_inseg, targets_inseg] = getbeat_pt_rjmartis(denoised, segment_labels, qrs_i_raw);
        if sum(targets_inseg) > 0
            targets(idx) = 1;
        end
        
        cA4 = peaks_inseg(:,1:107);
        cD4 = peaks_inseg(:,107+1: end);
        
        % PCA
        score_cA = cA4*coeff_cA;
        score_cD = cD4*coeff_cD;
        
        features = [score_cA(:,1:6) score_cD(:,1:6)];
        
        % SVM - RBF
        [predictions_inseg, ~] = predict(SVMModel, features);
        if sum(predictions_inseg) > 0
            predictions(idx) = 1;
        end
        
        idx = idx + 1;
    end
end

TP = 0;
FP = 0;
FN = 0;
TN = 0;
for i = 1:length(targets)
    if targets(i) == 0 && predictions(i) == 0
        TP = TP + 1;
    elseif targets(i) == 1 && predictions(i) == 0
        FP = FP + 1;
    elseif targets(i) == 0 && predictions(i) == 1
        FN = FN + 1;
    elseif targets(i) == 1 && predictions(i) == 1
        TN = TN + 1;
    else
        disp('Something Wrong!!!')
        break
    end
end
confusion_matrix = [TP, FP; FN, TN];
end