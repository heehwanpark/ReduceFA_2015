require 'torch'

torch.setnumthreads(16)

mlp_list = {{500}, {1000}, {500, 500}, {1000, 1000}, {500, 500, 500}}
feature_list = {'conv', 'max', 'min', 'max-min'}

-- doExperiment(trdata_type, testdata_type, mlp_architecture, feature_ex_type,
--             conv_architecture, conv_kernel, conv_pool, mwindow, db_seed,
--             net_init_seed, batchsize, lr, lr_decay, momentum, dropout_rate)

require 'doExperiment'
for i = 1, 5 do
  for j = 2, 4 do
    doExperiment('chal600+mimicAll', 'chal+mimic', mlp_list[i], feature_list[j])
  end
end
