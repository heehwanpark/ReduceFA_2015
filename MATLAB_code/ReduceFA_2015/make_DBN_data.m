% Making 10 sec training data
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
typeList = [];
for k=1:N
    fname = RECORDS{k};
    [~,signal,Fs,siginfo]=rdmat(fname);
    
    description=squeeze(struct2cell(siginfo));
    description=description(4,:);
    [dm, dn] = size(description);
    
    for j=1:dn
        n = length(description(j));
        if ~strncmp(typeList, description(j), n)
            typeList = [typeList, description(j)];
        end
    end    
end

%%
typeList = {'I', 'II', 'III', 'V', 'PLETH', 'RESP', 'aVF', 'ABP', 'MCL'};
countM = zeros(1,9);
for l=1:N
    fname = RECORDS{l};
    [~,signal,Fs,siginfo]=rdmat(fname);
    
    description=squeeze(struct2cell(siginfo));
    description=description(4,:);
    [ndm, ndn] = size(description);
    
    for j=1:ndn
        switch description{j}
            case typeList{1}
                countM(1) = countM(1)+1;
            case typeList{2}
                countM(2) = countM(2)+1;
            case typeList{3}
                countM(3) = countM(3)+1;
            case typeList{4}
                countM(4) = countM(4)+1;
            case typeList{5}
                countM(5) = countM(5)+1;
            case typeList{6}
                countM(6) = countM(6)+1;
            case typeList{7}
                countM(7) = countM(7)+1;                
            case typeList{8}
                countM(8) = countM(8)+1;
            case typeList{9}
                countM(9) = countM(9)+1;
        end
    end
end
disp(typeList)
disp(countM)

%%
% typeList = ['II', 'V', 'PLETH', 'RESP', 'aVF', 'ABP', 'MCL'];
% dataMatrix = zeros(750*30,2500*7+1);
% 
% count = 0;
% 
% fspecId = fopen('DBN_data\SPEC.txt','w');
% for i=1:N
%     fname = RECORDS{i};  
%     
%     [~, ~, ~, siginfo]=rdmat(fname);    
%     description=squeeze(struct2cell(siginfo));
%     description=description(4,:);
%     
%     struc = load(['training\' fname '.mat']);
%     val = struc.val;
%     for j=1:30
%         X = zeros(1,2500*7);
%         short_val = val(:,2500*(j-1)+1:2500*j);
%         for k=1:length(description)
%             switch description{k}
%                 case typeList(1)
%                     X(1:2500) = short_val(k,:);
%                 case typeList(2)
%                     X(2501:5000) = short_val(k,:);
%                 case typeList(3)
%                     X(5001:7500) = short_val(k,:);
%                 case typeList(4)
%                     X(7501:10000) = short_val(k,:);
%                 case typeList(5)
%                     X(10001:12500) = short_val(k,:);
%                 case typeList(6)
%                     X(12501:15000) = short_val(k,:);
%                 case typeList(7)
%                     X(15001:17500) = short_val(k,:);
%             end
%         end
%         
%         if j~=30
%             Y = 0;
%         else
%             if TF(i) == 1
%                 Y = 1;
%             else
%                 Y = 0;
%             end
%         end
%         
%         count = count + 1;
%         dataMatrix(count,:) = [X Y];
%         disp(count)
%     end
% end
% fclose(fspecId);

%%
pretraining = zeros(750*30,2500*3);
training = zeros(750,2500*3+1);
ptcount = 0;
tcount = 0;
for i=1:N
    fname = RECORDS{i};      
    [~, ~, ~, siginfo]=rdmat(fname);    
    description=squeeze(struct2cell(siginfo));
    description=description(4,:);    
    struc = load(['training\' fname '.mat']);
    val = struc.val;
    unluck = 0;
    for j=1:30
        X = zeros(1,2500*3);
        short_val = val(:,2500*(j-1)+1:2500*j);
        for k=1:length(description)
            switch description{k}
                case typeList(1)
                    X(1:2500) = short_val(k,:);
                    unluck = unluck + 1;
                case typeList(2)
                    X(2501:5000) = short_val(k,:);                    
                    unluck = unluck + 1;
                case typeList(3)
                    X(5001:7500) = short_val(k,:);                    
                    unluck = unluck + 1;
            end
        end
        
        if j == 30
            if TF(i) == 1
                Y = 1;
            else
                Y = 0;
            end         
            tcount = tcount + 1;  
            training(tcount,:) = [X Y];
        end
        
        ptcount = ptcount + 1;
        pretraining(ptcount,:) = X;
    end
    if unluck == 0
        disp(fname)
        disp(tcount)
    end
end
%%
save('data_10sec_dbn_pretraining.mat', 'pretraining')
save('data_10sec_dbn_training.mat', 'training')
csvwrite('data_10sec_dbn_pretraining.csv', pretraining)
csvwrite('data_10sec_dbn_training.csv', training)
