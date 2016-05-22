% Make dataset for DBN, only for ECG II

clc;
clear;

fid=fopen('/home/heehwan/Workspace/Data/ReduceFA_2015/training/ALARMS','r');
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

sec = 10;
inputlen = sec*250;
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
        struc = load(['/home/heehwan/Workspace/Data/ReduceFA_2015/training/' fname '.mat']);
        val = struc.val;        
        sel_val = val(idx,:); % +5 min long
        sel_val(isnan(sel_val)) = 0; % get rid of NaN
        recordlist{IIidx} = fname;
        IIidx = IIidx + 1;
        for j=1:num_segment
            X = sel_val(inputlen*(j-1)+1:inputlen*j);
            X = (X - mean(X))./std(X);
            X(isnan(X)) = 0;
            
            if j ~= num_segment
                ptcount = ptcount+1;
                pretraining(ptcount,:) = X;
            else
                tcount = tcount+1;
                training(tcount,:) = X;                
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

h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/chal2015.h5','/pretrain', size(pretraining));
h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/chal2015.h5', '/pretrain', pretraining);

h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/chal2015.h5','/inputs', size(training));
h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/chal2015.h5','/inputs', training);

h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/chal2015.h5','/targets', size(target));
h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/chal2015.h5','/targets', target);

save('recordlist.mat','recordlist');