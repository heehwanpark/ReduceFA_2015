require 'torch'

torch.setnumthreads(32)

-- 10/30
-- mlp_list = {{500}, {1000}, {500, 500}, {1000, 1000}, {500, 500, 500}}
-- feature_list = {'conv', 'max', 'min', 'max-min'}

-- 11/4
mlp_list = {{250, 250}, {500, 500}, {750, 750}, {1000, 1000}, {250, 250, 250}, {500, 500, 500}, {750, 750, 750}}
feature_list = {'mlp', 'max-min', 'max-min_x2', 'only_pool', 'only_pool_x2'}

-- doExperiment(trdata_type, testdata_type, mlp_architecture, feature_ex_type,
--             conv_architecture, conv_kernel, conv_pool, mwindow, db_seed,
--             net_init_seed, batchsize, lr, lr_decay, momentum, dropout_rate)

require 'doExperiment'

-- type 1
-- doExperiment('chal600+mimicAll', 'chal+mimic', {500}, 'conv')

-- type 2
-- for i = 1, table.getn(mlp_list) do
--   for j = 1, table.getn(feature_list) do
--     doExperiment('chal600+mimicAll', 'chal+mimic', mlp_list[i], feature_list[j])
--   end
-- end

-- type 3
for i = 1, table.getn(mlp_list) do
  doExperiment('chal600+mimicAll', 'chal+mimic', mlp_list[i], 'conv')
end

-- 11/25
-- db_seed_list = {1,2,3,5,7,11,13,17,19,23}
-- for i = 1, table.getn(db_seed_list) do
--   doExperiment('chal600+mimicAll', 'chal+mimic', mlp_list[3], 'max-min', nil, nil, nil, nil, db_seed_list[i])
--   doExperiment('chal600+mimicAll', 'chal+mimic', mlp_list[1], 'conv', nil, nil, nil, nil, db_seed_list[i])
-- end
