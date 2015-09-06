function [SVMModel, coeff_cA, coeff_cD] = training_rjmartis_seperated(datafolder, traininglist)
[N, ~] = size(traininglist);
peaks = zeros(120000, 107*2);
targets = zeros(120000, 1);
cursor = 1;
for i = 1:N
    filename = traininglist(i,:);
    disp(strcat('... ', filename))
    datafile = load(strcat(datafolder, filename, '.mat'));
    dataset = datafile.standard_dataset;
    ECG = dataset(1,:);
    labels = dataset(2,:);
    
    % Denoise ECG using by db6 wavelet transform
    denoised = denoising_rjmartis(ECG);
    
    % Pan-Tompkin QRS detection
    [~, qrs_i_raw, ~] = pan_tompkin(denoised, 360, false);
    
    [peaks_infile, targets_infile] = getbeat_pt_rjmartis(denoised, labels, qrs_i_raw);
    
    numpeaks = length(targets_infile);
    peaks(cursor:cursor+numpeaks-1, :) = peaks_infile;
    targets(cursor:cursor+numpeaks-1, :) = targets_infile;
    
    cursor = cursor + numpeaks;
end

peaks(cursor:end,:) = [];
targets(cursor:end,:) = [];

num_peaks = length(targets);
shuffled_idx = randperm(num_peaks);

peaks = peaks(shuffled_idx, :);
targets = targets(shuffled_idx);

cA4 = peaks(:,1:107);
cD4 = peaks(:,107+1: end);

% PCA
[coeff_cA, score_cA, ~] = pca(cA4, 'Centered', false);
[coeff_cD, score_cD, ~] = pca(cD4, 'Centered', false );

features = [score_cA(:,1:6) score_cD(:,1:6)];

% SVM - RBF
disp('...start learning SVM model')
SVMModel = fitcsvm(features, targets,'KernelFunction','rbf','Standardize',true);
end