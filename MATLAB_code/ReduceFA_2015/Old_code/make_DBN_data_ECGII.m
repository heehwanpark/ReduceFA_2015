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

sec = 10;
inputlen = 250 * sec;
num_segment = 300/sec;

pretraining = zeros(728*15,inputlen);
training = zeros(728,inputlen);
target = zeros(728,1);

ptcount = 0;
tcount = 0;

for i=1:N
    fname = RECORDS{i};      
    [~, ~, ~, siginfo]=rdmat(fname);
    description=squeeze(struct2cell(siginfo));
    description=description(4,:);
    flag = 0;
    if sum(strcmp(description,'II')) % II exists
        idx = find(strcmp(description,'II'));        
        struc = load(['training\' fname '.mat']);
        val = struc.val;        
        sel_val = val(idx,:); % +5 min long
        sel_val(isnan(sel_val)) = 0; 
        for j=1:15
            X = sel_val(inputlen*(j-1)+1:inputlen*j);
            X = (X - mean(X))./std(X);
            X(isnan(X)) = 0;
            ptcount = ptcount+1;
            pretraining(ptcount,:) = X;            
            if j == 15
                tcount = tcount+1;
                training(tcount,:) = X;                
                if TF(i) == 1 
                    Y = 1;
%                    alm = ALARMS{i};
%                    disp(alm)
%                    switch alm
%                        case 'Asystole'
%                            Y = 1;
%                        case 'Bradycardia'
%                            Y = 2;
%                        case 'Tachycardia'
%                            Y = 3;
%                        case 'Ventricular_Tachycardia'
%                            Y = 4;
%                        case 'Ventricular_Flutter_Fib'
%                            Y = 5;
%                    end
                else
                    Y = 0;
                end
                target(tcount) = Y;
            end
        end
    end    
end

h5create('DBN_data\data_normed_20sec_ECGII.h5','/pretrain', [728*15 inputlen]);
h5write('DBN_data\data_normed_20sec_ECGII.h5', '/pretrain', pretraining);

h5create('DBN_data\data_normed_20sec_ECGII.h5','/input', [728 inputlen]);
h5write('DBN_data\data_normed_20sec_ECGII.h5', '/input', training);

h5create('DBN_data\data_normed_20sec_ECGII.h5','/target', [728 1]);
h5write('DBN_data\data_normed_20sec_ECGII.h5', '/target', target);



% csvwrite('DBN_data\data_10sec_dbn_ECGII_pretraining_nan.csv', pretraining)
% csvwrite('DBN_data\data_10sec_dbn_ECGII_training_nan.csv', training)
% csvwrite('DBN_data\data_10sec_dbn_ECGII_target_nan.csv', target)