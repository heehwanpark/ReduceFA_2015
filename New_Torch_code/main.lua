require 'torch'

torch.setnumthreads(16)
trdata_type = 'chal600+mimicAll'
testdata_type = 'chal+mimic'
mlp_architecture = {500, 500}
feature_ex_type = 'max'
conv_architecture = nil
conv_kernel = nil
conv_pool = nil
mwindow = nil
db_seed = nil
net_init_seed = nil
batchsize = nil
lr = nil
lr_decay = nil
momentum = nil
dropout_rate = nil

require 'doExperiment'
doExperiment(trdata_type, testdata_type, mlp_architecture, feature_ex_type,
            conv_architecture, conv_kernel, conv_pool, mwindow, db_seed,
            net_init_seed, batchsize, lr, lr_decay, momentum, dropout_rate)
