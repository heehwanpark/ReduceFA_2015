require 'torch'
require 'cutorch'
require 'hdf5'
require 'nn'
require 'cunn'

torch.setnumthreads(16)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Reducing False Alarms in ICU')
cmd:text()
cmd:text('Options:')
-- Learning data
cmd:option('-trdata_type', 'chal600+mimicAll')
cmd:option('-testdata_type', 'chal+mimic')
-- Model
cmd:option('-inputSize', 2500) -- 250Hz * 10sec
cmd:option('-nInputFeature', 1)
---- Feature extraction type
cmd:option('-conv_architecture', {75, 75, 75})
cmd:option('-conv_kernel', 139)
cmd:option('-conv_pool', 2)
---- Experiment Setting
cmd:option('-db_seed', 1)
cmd:option('-net_init_seed', 1)
cmd:text()

option = cmd:parse(arg or {})
----------------------------------------------------------------------
local chal_file = hdf5.open('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5', 'r')
local chal_input = chal_file:read('/inputs'):all()
local chal_target = chal_file:read('/targets'):all()
chal_file:close()

chal_input = chal_input:transpose(1,2)
chal_target = chal_target:transpose(1,2)

torch.manualSeed(option.db_seed)
local nEle_chal = chal_target:size(1)
local shuffle_chal = torch.randperm(nEle_chal)
testset = torch.zeros(10, option.inputSize)
for i = 1, 10 do
  testset[{i, {}}] = chal_input[{shuffle_chal[i], {}}]
end
----------------------------------------------------------------------
foldername = '/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1109/'
art_init_model = torch.load(foldername .. 'art_init_model.net')
art_init_model = art_init_model:double()
art_weight_1 = art_init_model.modules[1].weight
art_weight_2 = art_init_model.modules[4].weight
art_weight_3 = art_init_model.modules[7].weight
----------------------------------------------------------------------
torch.manualSeed(option.net_init_seed)
model1 = nn.Sequential()

model1:add(nn.SpatialConvolutionMM(option.nInputFeature, option.conv_architecture[1], 1, option.conv_kernel))
model1:add(nn.ReLU())
model1:add(nn.SpatialMaxPooling(1, option.conv_pool))
nConvOut1 = math.floor((option.inputSize - option.conv_kernel + 1)/option.conv_pool)

model1.modules[1].weight = art_weight_1
----------------------------------------------------------------------
torch.manualSeed(option.net_init_seed)
model2 = nn.Sequential()

model2:add(nn.SpatialConvolutionMM(option.nInputFeature, option.conv_architecture[1], 1, option.conv_kernel))
model2:add(nn.ReLU())
model2:add(nn.SpatialMaxPooling(1, option.conv_pool))
nConvOut2 = math.floor((option.inputSize - option.conv_kernel + 1)/option.conv_pool)

model2.modules[1].weight = art_weight_1

model2:add(nn.SpatialConvolutionMM(option.conv_architecture[1], option.conv_architecture[2], 1, option.conv_kernel))
model2:add(nn.ReLU())
model2:add(nn.SpatialMaxPooling(1, option.conv_pool))
nConvOut2 = math.floor((nConvOut2 - option.conv_kernel + 1)/option.conv_pool)

model2.modules[4].weight = art_weight_2
----------------------------------------------------------------------
torch.manualSeed(option.net_init_seed)
model3 = nn.Sequential()

model3:add(nn.SpatialConvolutionMM(option.nInputFeature, option.conv_architecture[1], 1, option.conv_kernel))
model3:add(nn.ReLU())
model3:add(nn.SpatialMaxPooling(1, option.conv_pool))
nConvOut3 = math.floor((option.inputSize - option.conv_kernel + 1)/option.conv_pool)

model3.modules[1].weight = art_weight_1

model3:add(nn.SpatialConvolutionMM(option.conv_architecture[1], option.conv_architecture[2], 1, option.conv_kernel))
model3:add(nn.ReLU())
model3:add(nn.SpatialMaxPooling(1, option.conv_pool))
nConvOut3 = math.floor((nConvOut3 - option.conv_kernel + 1)/option.conv_pool)

model3.modules[4].weight = art_weight_2

model3:add(nn.SpatialConvolutionMM(option.conv_architecture[2], option.conv_architecture[3], 1, option.conv_kernel))
model3:add(nn.ReLU())
model3:add(nn.SpatialMaxPooling(1, option.conv_pool))
nConvOut3 = math.floor((nConvOut3 - option.conv_kernel + 1)/option.conv_pool)

model3.modules[7].weight = art_weight_3
----------------------------------------------------------------------
outputs1 = torch.zeros(10, 75, nConvOut1)
outputs2 = torch.zeros(10, 75, nConvOut2)
outputs3 = torch.zeros(10, 75, nConvOut3)

for t = 1, 10 do
  local input = testset[{{t, {}}}]
  input = torch.reshape(input, input:size(1), input:size(2), 1)

  output1 = model1:forward(input)
  output1 = torch.reshape(output1, output1:size(1), output1:size(2))
  outputs1[{t,{},{}}] = output1

  output2 = model2:forward(input)
  output2 = torch.reshape(output2, output2:size(1), output2:size(2))
  outputs2[{t,{},{}}] = output2

  output3 = model3:forward(input)
  output3 = torch.reshape(output3, output3:size(1), output3:size(2))
  outputs3[{t,{},{}}] = output3
end

test_record = hdf5.open(foldername .. 'conv_through.h5', 'w')
test_record:write('/inputs', testset)
test_record:write('/outputs1', outputs1)
test_record:write('/outputs2', outputs2)
test_record:write('/outputs3', outputs3)
test_record:close()
---------------------------------------------------------------------
