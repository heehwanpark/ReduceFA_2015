% Make dataset for DBN, only for ECG II

clc;
clear;

fid=fopen('training\ALARMS','r');
if(fid ~= -1)
    RECLIST=textscan(fid,'%s %s %d','Delimiter',',');
    fclose(fid);
else
    error('Could not open ALARMS.txt')
end

RECORDS=RECLIST{1};
ALARMS=RECLIST{2};
TF = RECLIST{3};
N=length(RECORDS);

oldfreq = 250;
newfreq = 360;
sec = 10;
segmentlen = oldfreq * sec;
inputlen = newfreq * sec;
num_segment = 300/sec;

pretraining = zeros(728*(num_segment-1),inputlen);
training = zeros(728,inputlen);
target = zeros(728,1);

ptcount = 0;
tcount = 0;

recordlist = cell(728,1);

IIidx = 1;
for i=1:N
    fname = RECORDS{i};
    [~, ~, ~, siginfo]=rdmat(fname);
    description=squeeze(struct2cell(siginfo));
    description=description(4,:);
    
    if sum(strcmp(description,'II')) % II exists
        idx = find(strcmp(description,'II'));        
        struc = load(['training\' fname '.mat']);
        val = struc.val;        
        sel_val = val(idx,:); % +5 min long
        sel_val(isnan(sel_val)) = 0; % get rid of NaN
        recordlist{IIidx} = fname;
        IIidx = IIidx + 1;
        for j=1:num_segment
            X = sel_val(segmentlen*(j-1)+1:segmentlen*j);
            X = (X - mean(X))./std(X);
            X(isnan(X)) = 0;
            
            % resampling
            n_X = downsample(interp(X, 36), 25);
            
            if j ~= num_segment
                ptcount = ptcount+1;
                pretraining(ptcount,:) = n_X;
            else
                tcount = tcount+1;
                training(tcount,:) = n_X;                
                if TF(i) == 1 
                    Y = 1;
                else
                    Y = 0;
                end
                target(tcount) = Y;
            end
        end
    end    
end

h5create('C:\Users\heehwan\workspace\Data\chal2015_data_10sec_resampled_0831.h5','/pretrain', [728*(num_segment-1) inputlen]);
h5write('C:\Users\heehwan\workspace\Data\chal2015_data_10sec_resampled_0831.h5', '/pretrain', pretraining);

h5create('C:\Users\heehwan\workspace\Data\chal2015_data_10sec_resampled_0831.h5','/input', [728 inputlen]);
h5write('C:\Users\heehwan\workspace\Data\chal2015_data_10sec_resampled_0831.h5', '/input', training);

h5create('C:\Users\heehwan\workspace\Data\chal2015_data_10sec_resampled_0831.h5','/target', [728 1]);
h5write('C:\Users\heehwan\workspace\Data\chal2015_data_10sec_resampled_0831.h5', '/target', target);

save('recordlist.mat','recordlist');