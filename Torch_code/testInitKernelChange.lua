require 'torch'
require 'custom'
require 'hdf5'
require 'nn'

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
cmd:option('-conv_architecture', 15)
cmd:option('-conv_kernel', 250)
cmd:option('-conv_pool', 2)
---- Experiment Setting
cmd:option('-db_seed', 1)
cmd:option('-net_init_seed', 1)
cmd:option('-initial_mode', false)
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
torch.manualSeed(option.net_init_seed)
model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(option.nInputFeature, option.conv_architecture, 1, option.conv_kernel))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(1, option.conv_pool))
----------------------------------------------------------------------
if option.initial_mode then
  foldername = '/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1106/'
  filename = 'KernelChange_init'
  test_record = hdf5.open(foldername .. filename .. '.h5', 'w')
  ----------------------------------------------------------------------
  k1 = model.modules[1].weight
  test_record:write('/init_weight', k1)
  ----------------------------------------------------------------------
  outputs = torch.zeros(10, option.conv_architecture, 1125)
  for t = 1, 10 do
    local input = testset[{{t, {}}}]
    input = torch.reshape(input, input:size(1), input:size(2), 1)
    output = model:forward(input)
    output = torch.reshape(output, output:size(1), output:size(2))
    outputs[{t,{},{}}] = output
  end
  test_record:write('/inputs', testset)
  test_record:write('/outputs_init', outputs)
  test_record:close()
else
  foldername = '/home/heehwan/Workspace/Data/ReduceFA_2015/revised_output/1106/'
  filename = 'KernelChange_new'
  test_record = hdf5.open(foldername .. filename .. '.h5', 'r')
  ----------------------------------------------------------------------
  new_weight = test_record:read('/new_weight'):all()
  test_record:close()
  new_weight = new_weight:transpose(1,2)
  model.modules[1].weight = new_weight
  ----------------------------------------------------------------------
  outputs = torch.zeros(10, option.conv_architecture, 1125)
  for t = 1, 10 do
    local input = testset[{{t, {}}}]
    input = torch.reshape(input, input:size(1), input:size(2), 1)
    output = model:forward(input)
    output = torch.reshape(output, output:size(1), output:size(2))
    outputs[{t,{},{}}] = output
  end
  test_record = hdf5.open(foldername .. filename .. '.h5', 'w')
  test_record:write('/inputs', testset)
  test_record:write('/outputs_new', outputs)
  test_record:close()
end
