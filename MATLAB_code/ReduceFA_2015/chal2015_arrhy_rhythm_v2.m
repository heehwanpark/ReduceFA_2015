% Make dataset

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
hz = 250;
input_size = sec*hz;
nSegment = 300/sec;
nChannel = 4;

for i=1:N
    fname = RECORDS{i};
    [~, ~, ~, siginfo]=rdmat(fname);
    description=squeeze(struct2cell(siginfo));
    description=description(4,:);

    inputs = zeros(nChannel, input_size);

    struc = load(['/home/heehwan/Workspace/Data/ReduceFA_2015/training/' fname '.mat']);
    val = struc.val; 
    val(isnan(val)) = 0;

    if any(strcmp(description,'II'))
        idx_2 = find(strcmp(description,'II'));
        val_2 = val(idx_2,72501:75000);
        val_2 = (val_2-mean(val_2))./std(val_2);
        inputs(1,:) = val_2;
    end

    if any(strcmp(description, 'V'))
        idx_5 = find(strcmp(description,'V'));
        val_5 = val(idx_5,72501:75000);
        val_5 = (val_5-mean(val_5))./std(val_5);
        inputs(2,:) = val_5;
    end

    if any(strcmp(description, 'PLETH'))
        idx_P = find(strcmp(description,'PLETH'));
        val_P = val(idx_P,72501:75000);
        val_P = (val_P-mean(val_P))./std(val_P);
        inputs(3,:) = val_P;
    end

    if any(strcmp(description, 'ABP'))
        idx_A = find(strcmp(description,'ABP'));
        val_A = val(idx_A,72501:75000);
        val_A = (val_A-mean(val_A))./std(val_A);
        inputs(4,:) = val_A;
    end
    
    if ~any(strcmp(description,'II')) && ~any(strcmp(description, 'V')) && ~any(strcmp(description, 'PLETH')) && ~any(strcmp(description, 'ABP'))
        disp(fname)
    end
    
    if TF(i) == 1
        target = 1;
    else
        target = 0;
    end
    
    h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/chal2015_v2.h5',['/' int2str(i) '/inputs'], size(inputs));
    h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/chal2015_v2.h5',['/' int2str(i) '/inputs'], inputs);
    
    h5create('/home/heehwan/Workspace/Data/ReduceFA_2015/chal2015_v2.h5',['/' int2str(i) '/target'], 1);
    h5write('/home/heehwan/Workspace/Data/ReduceFA_2015/chal2015_v2.h5',['/' int2str(i) '/target'], target);
end