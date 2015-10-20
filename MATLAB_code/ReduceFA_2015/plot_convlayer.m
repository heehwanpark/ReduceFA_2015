clc;
clear;

chal600_conv_01 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_02/04_convlayer/kernel250/chal600+mimicAll.h5','/conv1_weight');
chal600_conv_02 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_02/04_convlayer/kernel250/chal600+mimicAll.h5','/conv2_weight');
chal600_conv_03 = h5read('/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/experiment_02/04_convlayer/kernel250/chal600+mimicAll.h5','/conv3_weight');

layer = 1;
if layer == 1
    chal600_c01_original = zeros(75,250);
    chal600_c01_updated = zeros(75,250);
    for i = 1:75
        kernel_origin = squeeze(chal600_conv_01(:,i,1));
        kernel_origin = kernel_origin';
        kernel_updated = squeeze(chal600_conv_01(:,i,2));
        kernel_updated = kernel_updated';
        chal600_c01_original(i,:) = kernel_origin;
        chal600_c01_updated(i,:) = kernel_updated;
    end
    num_in_fig = 2;
    for j = 1:floor(75/num_in_fig)+1
        figure
        x = 1:250;
        if 75-num_in_fig*j >= 0 
            for k = 1:num_in_fig
                subplot(2,1,k)
                idx = (j-1)*num_in_fig + k;
                plot(x,chal600_c01_original(idx,:),'-o', x,chal600_c01_updated(idx,:),'-+')
            end
        else
            for k = 1:75-num_in_fig*(j-1)
                subplot((num_in_fig/2),2,k)
                idx = (j-1)*num_in_fig + k;
                plot(x,chal600_c01_original(idx,:),'-o', x,chal600_c01_updated(idx,:),'-+')
            end
        end
    end
elseif layer == 2
    chal600_c02_original = zeros(75,250);
    chal600_c02_updated = zeros(75,250);
    
    kernel_origin = squeeze(chal600_conv_02(:,1,1));
    kernel_origin = kernel_origin';
    kernel_updated = squeeze(chal600_conv_02(:,1,2));
    kernel_updated = kernel_updated';    
    for i = 1:75
        s_i = (i-1)*250+1;
        e_i = s_i+250-1;
        chal600_c02_original(i,:) = kernel_origin(s_i:e_i);
        chal600_c02_updated(i,:) = kernel_updated(s_i:e_i);
    end
    num_in_fig = 2;
    for j = 1:floor(75/num_in_fig)+1
        figure
        x = 1:250;
        if 75-num_in_fig*j >= 0 
            for k = 1:num_in_fig
                subplot(2,1,k)
                idx = (j-1)*num_in_fig + k;
                plot(x,chal600_c02_original(idx,:),'-o', x,chal600_c02_updated(idx,:),'-+')
            end
        else
            for k = 1:75-num_in_fig*(j-1)
                subplot((num_in_fig/2),2,k)
                idx = (j-1)*num_in_fig + k;
                plot(x,chal600_c02_original(idx,:),'-o', x,chal600_c02_updated(idx,:),'-+')
            end
        end
    end
elseif layer == 3
    chal600_c03_original = zeros(75,250);
    chal600_c03_updated = zeros(75,250);
    
    kernel_origin = squeeze(chal600_conv_03(:,1,1));
    kernel_origin = kernel_origin';
    kernel_updated = squeeze(chal600_conv_03(:,1,2));
    kernel_updated = kernel_updated';    
    for i = 1:75
        s_i = (i-1)*250+1;
        e_i = s_i+250-1;
        chal600_c03_original(i,:) = kernel_origin(s_i:e_i);
        chal600_c03_updated(i,:) = kernel_updated(s_i:e_i);
    end
    num_in_fig = 2;
    for j = 1:floor(75/num_in_fig)+1
        figure
        x = 1:250;
        if 75-num_in_fig*j >= 0 
            for k = 1:num_in_fig
                subplot(2,1,k)
                idx = (j-1)*num_in_fig + k;
                plot(x,chal600_c03_original(idx,:),'-o', x,chal600_c03_updated(idx,:),'-+')
            end
        else
            for k = 1:75-num_in_fig*(j-1)
                subplot((num_in_fig/2),2,k)
                idx = (j-1)*num_in_fig + k;
                plot(x,chal600_c03_original(idx,:),'-o', x,chal600_c03_updated(idx,:),'-+')
            end
        end
    end
end


