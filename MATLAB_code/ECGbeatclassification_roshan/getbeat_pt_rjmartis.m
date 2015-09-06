function [peaks, targets] = getbeat_pt_rjmartis(denoised, labels, qrs_i_raw)    
    N = length(qrs_i_raw);    
    peaks = zeros(N, 107*2);
    targets = zeros(N, 1);
    
    cood_labels = find(labels ~= 0);
    index = 1;
    for j = 1:N
        Rpoint = qrs_i_raw(j);
        if Rpoint >= 100 && Rpoint <= (length(denoised)-100)
            start_idx = Rpoint-99;
            end_idx = Rpoint+100;
            QRSpeak = denoised(start_idx:end_idx);
            
            [c_dmey, l_dmey] = wavedec(QRSpeak, 4, 'dmey');
            cA4_cD4 = c_dmey(1:(l_dmey(1)+l_dmey(2)));
            peaks(index, :) = cA4_cD4;
                      
            normality = 0;
            for k = 1:length(cood_labels)
                if cood_labels(k) >= start_idx && cood_labels(k) <= end_idx
                    if labels(cood_labels(k)) == 2
                        normality = 1;
                    end
                end
            end            
            targets(index) = normality;            
            
            index = index + 1;
        end
    end
    peaks(index:end,:) = [];
    targets(index:end,:) = [];
end