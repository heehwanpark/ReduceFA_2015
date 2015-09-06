% Making standard mitdb dataset for conventional approach and new approach
% (CNN)

clc;
clear;

datafolder = 'C:\Users\heehwan\workspace\Data\MIT_BIH\';
filelist = ['100m'; '101m'; '102m'; '103m'; '104m'; '105m'; '106m'; '107m'; '108m'; '109m';
                '111m'; '112m'; '113m'; '114m'; '115m'; '116m'; '117m'; '118m'; '119m'; '121m';
                '122m'; '123m'; '124m'; '200m'; '201m'; '202m'; '203m'; '205m'; '207m'; '208m';
                '209m'; '210m'; '212m'; '213m'; '214m'; '215m'; '217m'; '219m'; '220m'; '221m';
                '222m'; '223m'; '228m'; '230m'; '231m'; '232m'; '233m'; '234m'];
            
for i = 1:48
    filename = filelist(i,:);
    filenum = filename(1:3);
    disp(filename);
    
    header = fopen(strcat(datafolder, filename, '.hea'));
    C = textscan(header, '%s %s %s %d %d %d %d %d %s', 2, 'headerLines', 1);
    types = C{9};
    
    isII = 0;
    for j = 1:2
        if strcmp(types{j}, 'MLII')
            isII = 1;
            col_idx = j;
        end
    end
    if isII
        ecgfile = load(strcat(datafolder, filename, '.mat'));
        dataset = ecgfile.val;
        ECG = dataset(col_idx,:);
        len = length(ECG);  

        [cood, annot] = getAnnotation(filenum, datafolder);    

        standard_dataset = zeros(2,len);
        standard_dataset(1,:) = ECG;
        standard_dataset(2,cood) = annot;

        savefolder = 'C:\Users\heehwan\workspace\Data\Standard\mitdb\';
        savefile = strcat(savefolder, filename);
        % mat file
        save(strcat(savefile,'.mat'), 'standard_dataset');
        % hdf5 file
        h5create(strcat(savefile,'.h5'),'/dataset', [2 len]);
        h5write(strcat(savefile,'.h5'),'/dataset', standard_dataset);
    else
        disp('There is no MLII signal')
        disp(filename)
    end
end