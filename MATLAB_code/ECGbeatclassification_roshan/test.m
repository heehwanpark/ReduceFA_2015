% Reproduce Roshan's work
clc;
clear;

% % LTFAT setup
% cd 'C:\Program Files\MATLAB\R2014b\toolbox\ltfat'
% ltfatstart
% 
% % Come back
% cd 'C:\Users\heehwan\workspace\MATLAB\ECGbeatclassification_roshan'

datafolder = 'C:\Users\heehwan\workspace\Data\MIT_BIH\';
filelist = ['100m'; '101m'; '102m'; '103m'; '104m'; '105m'; '106m'; '107m'; '108m'; '109m';
            '111m'; '112m'; '113m'; '114m'; '115m'; '116m'; '117m'; '118m'; '119m'; '121m';
            '122m'; '123m'; '124m'; '200m'; '201m'; '202m'; '203m'; '205m'; '207m'; '208m';
            '209m'; '210m'; '212m'; '213m'; '214m'; '215m'; '217m'; '219m'; '220m'; '221m';
            '222m'; '223m'; '228m'; '230m'; '231m'; '232m'; '233m'; '234m'];

whole_peaks = zeros(120000, 200);
cursor = 1;
for j=1:48
    filename = filelist(j,:);
    ecgfile = load(strcat(datafolder, filename, '.mat'));
    dataset = ecgfile.val;
    ECG = dataset(1,:);

    % Denoise
    [C, L] = wavedec(ECG, 9, 'db6');
    remains = sum(L(1:8));
    nc = C;
    nc(remains+1:end) = 0;
    den_ECG2 = waverec(nc, L, 'db6');

    % QRS
    [qrs_amp_raw,qrs_i_raw,delay] = pan_tompkin(den_ECG2, 360, false);
    for i = qrs_i_raw    
        if i >= 100 && (650000-i) >= 100
            start_idx = i-99;
            end_idx = i+100;
            QRSpeak = den_ECG2(start_idx:end_idx);
            whole_peaks(cursor, :) = QRSpeak;
            cursor = cursor+1;
        end
    end
end
whole_peaks(cursor:end, :) = [];

% PCA
