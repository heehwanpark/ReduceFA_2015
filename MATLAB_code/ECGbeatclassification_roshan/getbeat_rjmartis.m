function [peaks, targets] = getbeat_rjmartis(denoised, labels)
    cood_labels = find(labels ~= 0);
    N = length(cood_labels);
%     N = length(qrs_i_raw);    
    peaks = zeros(N, 107*2);
    targets = zeros(N, 1);
    
    index = 1;
%     label_cursor = 1;
    for j = 1:N
        Rpoint = cood_labels(j);
        if Rpoint >= 100 && Rpoint <= (length(denoised)-100)
            start_idx = Rpoint-99;
            end_idx = Rpoint+100;
            QRSpeak = denoised(start_idx:end_idx);
            
            [c_dmey, l_dmey] = wavedec(QRSpeak, 4, 'dmey');
            cA4_cD4 = c_dmey(1:(l_dmey(1)+l_dmey(2)));
            peaks(index, :) = cA4_cD4;
            
            targets(index) = labels(cood_labels(j)) - 1;            
%             normality = 0;
%             while label_cursor <= length(cood_labels) && cood_labels(label_cursor) >= start_idx && cood_labels(label_cursor) <= end_idx
%                 if labels(cood_labels(label_cursor)) == 2
%                     normality = 1;
%                 end
%                 label_cursor = label_cursor + 1;
%             end
%             targets(index) = normality;            
            
            index = index + 1;
        end
    end
    peaks(index:end,:) = [];
    targets(index:end,:) = [];
end