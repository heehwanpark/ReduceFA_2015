% load feature data
% [65 cA, 65 cD]

feature_data = h5read('C:\Users\heehwan\workspace\Data\MIT_BIH\wholefeatures.h5', '/features');
feature_data = feature_data';

cA = feature_data(:,1:65);
cD = feature_data(:,66:end);

[coeff, score, latent] = pca(cA);
