clc;
clear;

fullinputdata = hdf5read('mitbih_data_10sec_v1.h5', '/inputs');
fullinputdata = fullinputdata';

fulltargetdata = hdf5read('mitbih_data_10sec_v1.h5', '/targets');
fulltargetdata = fulltargetdata';

pair_list = hdf5read('faultfile_chal.h5', '/list');
pair_list = pair_list';

[m, n] = size(pair_list);

FP = [];
FN = [];
for i = 1:m
    if pair_list(i,2) == 0
        index = pair_list(i,1);
        if fulltargetdata(index) == 1
            FP = [FP; index];
        else
            FN = [FN; index];
        end
    end
end

record_list = [100 101 103 105 106 107 108 109 111 112 113 114 115 116 117 118 119 121 122 123 124 200 201 202 203 205 207 208 209 210 212 213 214 215 217 219 220 221 222 223 228 230 231 232 233 234];
FP = sort(FP);

for i = 1:length(FP)
    index = FP(i);
    dbname = record_list(floor(index/180)+1);
    location = (mod(index, 180)-1)*10;
    disp('-------------------------------')
    disp(dbname)
    min = num2str(floor(location/60));
    sec = num2str(mod(location, 60));
    time = strcat(min,':',sec);
    disp(time)
end
