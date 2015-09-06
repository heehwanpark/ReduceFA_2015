function [coordinate, annotation] = getAnnotation(file_num, data_folder)
    ann_file = strcat(data_folder,file_num,'.txt');
    fid = fopen(ann_file);
    C = textscan(fid, '%s %d %s %d %d %d %s', -inf, 'headerLines', 1);
    fclose(fid);
    
    C_3 = C{3};
    len = length(C_3);
    class_list = zeros(len,1);
    plus_list = zeros(len,1);
    for i=1:len
        ann = C_3{i};
        switch ann
            case {'N','L','R','B','e','j'}
                class = 1;
            case {'A','a','J','S'}
                class = 2;
            case {'V','r','n','E'}
                class = 2;
            case {'F'}
                class = 2;
            case {'f','Q','/','?'}
                class = 2;
            otherwise
                plus_list(i) = i;
                continue
        end
        class_list(i) = class;
    end
    plus_list(plus_list == 0) = [];
    
    coordinate = C{2};
    coordinate(plus_list) = [];
    annotation = class_list;
    annotation(plus_list) = [];
end