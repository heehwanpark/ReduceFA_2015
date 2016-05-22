require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'hdf5'

-- foldername = '/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1109/'
-- file = hdf5.open(foldername .. 'kernelFromModel.h5', 'w')
--
-- art_init_model = torch.load('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1109/art_init_model.net')
-- art_init_model = art_init_model:double()
-- art_weight_1 = art_init_model.modules[1].weight
-- art_weight_2 = art_init_model.modules[4].weight
-- art_weight_3 = art_init_model.modules[7].weight
--
-- file:write('/art_weight_1', art_weight_1)
-- file:write('/art_weight_2', art_weight_2)
-- file:write('/art_weight_3', art_weight_3)
--
-- rand_init_model = torch.load('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1109/rand_init_model.net')
-- rand_init_model = art_init_model:double()
-- rand_weight_1 = rand_init_model.modules[1].weight
-- rand_weight_2 = rand_init_model.modules[4].weight
-- rand_weight_3 = rand_init_model.modules[7].weight
--
-- file:write('/rand_weight_1', rand_weight_1)
-- file:write('/rand_weight_2', rand_weight_2)
-- file:write('/rand_weight_3', rand_weight_3)
--
-- file:close()

-- 11/26
foldername = '/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1125/'
file = hdf5.open(foldername .. 'kernelFromModel.h5', 'w')

init_model = torch.load('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1125/[250-250]-false-conv-seed-1-initial_model.net')
init_model = init_model:double()
init_weight_1 = init_model.modules[1].weight
init_weight_2 = init_model.modules[4].weight
init_weight_3 = init_model.modules[7].weight

file:write('/init_weight_1', init_weight_1)
file:write('/init_weight_2', init_weight_2)
file:write('/init_weight_3', init_weight_3)

train_model = torch.load('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1125/[250-250]-false-conv-seed-1-trained_model.net')
train_model = train_model:double()
train_weight_1 = train_model.modules[1].weight
train_weight_2 = train_model.modules[4].weight
train_weight_3 = train_model.modules[7].weight

file:write('/train_weight_1', train_weight_1)
file:write('/train_weight_2', train_weight_2)
file:write('/train_weight_3', train_weight_3)

file:close()
