function cell_cmatrix = traintest_rjmartis_original(datafolder, filelist)
    if exist('peaks_pt.mat', 'file') == 2
        load('peaks_pt.mat');
        load('targets_pt.mat');
    else
        [nRecord, ~] = size(filelist);    
        peaks = zeros(120000, 107*2);
        targets = zeros(120000, 1);
        cursor = 1;
        for i = 1:nRecord
            filename = filelist(i,:);
            disp(strcat('... ', filename))
            datafile = load(strcat(datafolder, filename, '.mat'));
            dataset = datafile.standard_dataset;
            ECG = dataset(1,:);
            labels = dataset(2,:);

            % Denoise ECG using by db6 wavelet transform
            denoised = denoising_rjmartis(ECG);

            % Pan-Tompkin QRS detection
            [~, qrs_i_raw, ~] = pan_tompkin(denoised, 360, false);

%             [peaks_infile, targets_infile] = getbeat_rjmartis(denoised, labels);
            [peaks_infile, targets_infile] = getbeat_pt_rjmartis(denoised, labels, qrs_i_raw);

            numpeaks = length(targets_infile);
            peaks(cursor:cursor+numpeaks-1, :) = peaks_infile;
            targets(cursor:cursor+numpeaks-1, :) = targets_infile;

            cursor = cursor + numpeaks;
        end
        peaks(cursor:end, :) = [];
        targets(cursor:end, :) = [];
        
        save('peaks_pt.mat', 'peaks');
        save('targets_pt.mat', 'targets');
    end
    nPeaks = length(targets);
    nTraining = floor(nPeaks*0.9);
    nTesting = nPeaks - nTraining;
    
    shuffled_idx = randperm(nPeaks);
    cell_cmatrix = cell(10);
    for j = 1:10
        disp(j)
        disp('fold')
        if j == 1
            test_idx = shuffled_idx(1:nTesting);
            training_idx = shuffled_idx(nTesting+1:end);
        elseif j == 10
            test_idx = shuffled_idx(nTraining+1:end);
            training_idx = shuffled_idx(1:nTraining);
        else
            s_i = nTesting*(j-1)+1;
            e_i = s_i+nTesting-1;
            test_idx = shuffled_idx(s_i:e_i);
            training_idx = [shuffled_idx(1:s_i-1) shuffled_idx(e_i+1:end)];
        end
        
        training_inputs = peaks(training_idx, :);
        training_targets = targets(training_idx);
        
        testing_inputs = peaks(test_idx, :);
        testing_targets = targets(test_idx);
        
        %%%%%%%%%%%%
        % Training %
        %%%%%%%%%%%%
        
        cA4_tr = training_inputs(:, 1:107);
        cD4_tr = training_inputs(:, 108:end);
        
        % PCA
        [coeff_cA, score_cA, latent_cA] = pca(cA4_tr, 'Centered', false);
        [coeff_cD, score_cD, latent_cD] = pca(cD4_tr, 'Centered', false );
        
        features = [score_cA(:,1:6) score_cD(:,1:6)];
        
        % SVM - RBF
        SVMModel = fitcsvm(features, training_targets,'KernelFunction','rbf','Standardize',true);
        
        %%%%%%%%%%%
        % Testing %
        %%%%%%%%%%%
        
        cA4_te = testing_inputs(:, 1:107);
        cD4_te = testing_inputs(:, 108:end);
        
        score_cA_te = cA4_te*coeff_cA;
        score_cD_te = cD4_te*coeff_cD;
        
        test_features = [score_cA_te(:,1:6) score_cD_te(:,1:6)];
        
        [predictions, ~] = predict(SVMModel, test_features);
        
        TP = 0;
        FP = 0;
        FN = 0;
        TN = 0;
        for i = 1:nTesting
            if testing_targets(i) == 0 && predictions(i) == 0
                TP = TP + 1;
            elseif testing_targets(i) == 1 && predictions(i) == 0
                FP = FP + 1;
            elseif testing_targets(i) == 0 && predictions(i) == 1
                FN = FN + 1;
            elseif testing_targets(i) == 1 && predictions(i) == 1
                TN = TN + 1;
            else
                disp('Something Wrong!!!')
                break
            end
        end
        confusion_matrix = [TP FP; FN TN];
        cell_cmatrix{j} = confusion_matrix;
    end
end