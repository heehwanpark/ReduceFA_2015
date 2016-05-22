clc;
clear;

datafolder = '/media/heehwan/HDD_1TB/WFDB_data/MIT_BIH/';
filelist = {'100m', '101m', '103m', '105m', '106m', '107m', '108m', '109m', ...
            '111m', '112m', '113m', '114m', '115m', '116m', '117m', '118m', '119m', '121m', ...
            '122m', '123m', '124m', '200m', '201m', '202m', '203m', '205m', '207m', '208m', ...
            '209m', '210m', '212m', '213m', '214m', '215m', '217m', '219m', '220m', '221m', ...
            '222m', '223m', '228m', '230m', '231m', '232m', '233m', '234m'};

% selected beat
beats_100m = {370}; beats_101m = {396}; beats_103m = {575}; beats_105m = {387781};
beats_106m = {37123}; beats_107m = {32111}; beats_108m = {4105}; beats_109m = {6259};
beats_111m = {489}; beats_112m = {382}; beats_113m = {583}; beats_114m = {107503};
beats_115m = {518}; beats_116m = {35132}; beats_117m = {598}; beats_118m = {369};
beats_119m = {503}; beats_121m = {513}; beats_122m = {561}; beats_123m = {152471};
beats_124m = {108189, 108933}; beats_200m = {689}; beats_201m = {420680}; beats_202m = {417763}; 
beats_203m = {16726}; beats_205m = {88985}; beats_207m = {14522, 15906}; beats_208m = {14619};
beats_209m = {618349}; beats_210m = {625}; beats_212m = {1440}; beats_213m = {478};
beats_214m = {3235}; beats_215m = {10490}; beats_217m = {812}; beats_219m = {614};
beats_220m = {916}; beats_221m = {7689}; beats_222m = {646}; beats_223m = {520287};
beats_228m = {6048}; beats_230m = {13208}; beats_231m = {847}; beats_232m = {130614};
beats_233m = {1107}; beats_234m = {1327};

beats_list = zeros(48, 139);
index = 1;
for i = 1:length(filelist)
    beats = 'beats_';
    file = filelist{i};
    ecg = load([datafolder file '.mat']);
    ecg = ecg.val;
    ecgII = ecg(1,:);
    beats_file = eval([beats file]);
    for j = 1:length(beats_file)
        Rpoint = beats_file{j};
        beat_start = Rpoint-99;
        beat_end = Rpoint+100;
        beat = ecgII(beat_start:beat_end);
        beat = downsample(interp(beat,25),36);
        beats_list(index,:) = beat;
        index = index+1;
    end
end

% h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1106/artificial_kernel_v1.h5', '/weight', size(beats_list));
% h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1106/artificial_kernel_v1.h5', '/weight', beats_list);