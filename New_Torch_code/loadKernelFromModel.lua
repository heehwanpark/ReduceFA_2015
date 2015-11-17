require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'hdf5'

foldername = '/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1109/'
file = hdf5.open(foldername .. 'kernelFromModel.h5', 'w')

art_init_model = torch.load('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1109/art_init_model.net')
art_init_model = art_init_model:double()
art_weight_1 = art_init_model.modules[1].weight
art_weight_2 = art_init_model.modules[4].weight
art_weight_3 = art_init_model.modules[7].weight

file:write('/art_weight_1', art_weight_1)
file:write('/art_weight_2', art_weight_2)
file:write('/art_weight_3', art_weight_3)

rand_init_model = torch.load('/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1109/rand_init_model.net')
rand_init_model = art_init_model:double()
rand_weight_1 = rand_init_model.modules[1].weight
rand_weight_2 = rand_init_model.modules[4].weight
rand_weight_3 = rand_init_model.modules[7].weight

file:write('/rand_weight_1', rand_weight_1)
file:write('/rand_weight_2', rand_weight_2)
file:write('/rand_weight_3', rand_weight_3)

file:close()
