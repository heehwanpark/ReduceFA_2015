require 'torch'
require 'custom_HH'
----------------------------------------------------------------------
print '==> Processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Reducing False Alarms using CNNs')
cmd:text()
cmd:text('Options:')
-- Data
cmd:option('-data', '/home/heehwan/Documents/workspace/data/ReduceFA/data_normed_20sec_ECGII.h5')
-- Model
-- Label: normal = 0, Asystole = 1, Bradycardia = 2, Tachycardia = 3,
-- Ventricular Tachycardia = 4, Ventricular Flutter/Fibrillation = 5
cmd:option('-nTarget', 2)
cmd:option('-nInputFeature', 1)
cmd:option('-inputSize', 5000) -- 250Hz * 20sec
--- For convolutional networks
cmd:option('-nFeatures1', 16)
cmd:option('-nFeatures2', 64)
--- For MLP
cmd:option('-nFeatures3', 1024)
--- For PSD
cmd:option('-lambda', 1)
cmd:option('-beta', 1)
-- Experiment Setting
cmd:option('-seed', 1)
cmd:option('-batchsize', 10)
cmd:option('-nFold', 10)
cmd:option('-maxIter', 100)
cmd:option('-lr_sup', 0.01, 'Learning rate')
cmd:option('-lr_unsup', 1e-5, 'Learning rate')
cmd:option('-lrdecay',1e-4, 'Learning rate decay')
cmd:option('-momentum', 0)
cmd:option('-pretraining', true)
-- Conv Setting
cmd:option('-kernel', 10)
cmd:option('-pool', 4)
-- Torch Setting
cmd:option('-thread', 16)

cmd:text()
option = cmd:parse(arg)
----------------------------------------------------------------------
print '==> Setting'

torch.manualSeed(option.seed)
torch.setnumthreads(option.thread)
----------------------------------------------------------------------
print '==> Load datasets'

require 'hdf5'
local datafile = hdf5.open(option.data, 'r')
local pretrainset = datafile:read('/pretrain'):all()
local labelset_input = datafile:read('/input'):all()

local labelset_target = datafile:read('/target'):all()
datafile:close()

pretrainset = pretrainset:transpose(1,2)
labelset_input = labelset_input:transpose(1,2)
labelset_target = labelset_target:transpose(1,2)

----------------------------------------------------------------------
require 'optim'

pretrainset1 = convertForPretrain(pretrainset)

if option.pretraining then
  require 'unsup'
  require 'ConvPSD_HH'
  -- 1st layer
  -- encoder1, decoder1 = trainConvPSD(pretrainset1, option.nInputFeature, option.nFeatures1, option, 'pretrain_result_layer1')
  encoder1 = torch.load('pretrain_result_layer1_encoder.net')
  decoder1 = torch.load('pretrain_result_layer1_decoder.net')
  -- pretrainset2 = netsThrough(encoder1, pretrainset1)
  -- -- 2nd layer
  -- encoder2, decoder2 = trainConvPSD(pretrainset2, option.nFeatures1, option.nFeatures2, option, 'pretrain_result_layer2')
end
----------------------------------------------------------------------
-- TODO: should we initialize imported module?
require 'cutorch'
require 'cunn'

print '==> Construct CNN model'
print '==> construct model'

model = nn.Sequential()

-- 1st convolution layer
model:add(nn.SpatialConvolutionMM(option.nInputFeature, option.nFeatures1, 1, option.kernel))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(1, option.pool))

-- Calculate # of outputs
nConvOut = math.floor((option.inputSize - option.kernel + 1)/option.pool)

-- 2nd convolution layer
model:add(nn.SpatialConvolutionMM(option.nFeatures1, option.nFeatures2, 1, option.kernel))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(1, option.pool))

-- Calculate # of outputs
nConvOut = math.floor((nConvOut - option.kernel + 1)/option.pool)

-- Standard MLP
model:add(nn.View(option.nFeatures2*nConvOut*1))
model:add(nn.Linear(option.nFeatures2*nConvOut*1, option.nFeatures3))
model:add(nn.ReLU())
model:add(nn.Linear(option.nFeatures3, option.nFeatures3))
model:add(nn.ReLU())
-- model:add(nn.Dropout(0.25))
model:add(nn.Linear(option.nFeatures3, option.nTarget))
model:add(nn.LogSoftMax())

if option.pretraining then
  model.modules[1].weight = encoder1.modules[1].weight
  -- model[4].weight = encoder2.modules[1].weight
end

model:cuda()

print(model)

----------------------------------------------------------------------
print '==> Defining loss'

weight = torch.Tensor(2)
weight[1] = 0.38
weight[2] = 0.62
criterion = nn.ClassNLLCriterion(weight)
-- criterion = nn.ClassNLLCriterion()
criterion:cuda()
----------------------------------------------------------------------
print '==> Defining some tools'

classes = {'Normal','Abnormal'}

confusion = optim.ConfusionMatrix(classes)
----------------------------------------------------------------------
print '==> configuring optimizer'
optimState = {
  learningRate = option.lr_sup,
  weightDecay = 0,
  momentum = 0,
  learningRateDecay = option.lrdecay
}
optimMethod = optim.sgd

----------------------------------------------------------------------
print '==> Defining training procedure'

parameters, gradParameters = model:getParameters()

batchsize = option.batchsize

nFold = 10
nElement = labelset_target:size(1)
nTraining = nElement - math.floor(nElement/nFold)
nTesting = nElement - nTraining

trainset_input = labelset_input[{{1,nTraining}, {}}]
trainset_target = labelset_target[{{1,nTraining}, {}}]

testset_input = labelset_input[{{nTraining+1, nElement}, {}}]
testset_target = labelset_target[{{nTraining+1, nElement}, {}}]

result_train_accu = torch.zeros(100)
result_train_err = torch.zeros(100)

result_test_accu = torch.zeros(100)
result_test_err = torch.zeros(100)

function train()
  epoch = epoch or 1

  cur_err = 0

  local time = sys.clock()

  model:training()

  shuffle = torch.randperm(nTraining)

  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchsize .. ']')
  for t = 1, math.floor(nTraining/batchsize) do
    local inputs = {}
    local targets = {}
    batchstart = (t-1)*batchsize+1
    for i = batchstart,batchstart+batchsize-1 do
      local input = trainset_input[{{shuffle[i], {}}}]
      input = torch.reshape(input, input:size(1), input:size(2), 1):cuda()
      local target = trainset_target[shuffle[i]][1]+1
      table.insert(inputs, input)
      table.insert(targets, target)
    end

    local feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end

      gradParameters:zero()

      local f = 0
      for i = 1, #inputs do
        local output = model:forward(inputs[i])
        local err = criterion:forward(output, targets[i])
        f = f + err
        local df_do = criterion:backward(output, targets[i])
        model:backward(inputs[i], df_do)

        confusion:add(output, targets[i])
      end

      gradParameters:div(#inputs)
      f = f/#inputs
      return f, gradParameters
    end

    _,fs = optimMethod(feval, parameters, optimState)
    cur_err = cur_err + fs[1]
  end

  time = sys.clock() - time
  time = time / nTraining

  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

  print(confusion)
  cur_err = cur_err/math.floor(nTraining/batchsize)

  result_train_accu[epoch] = confusion.totalValid
  result_train_err[epoch] = cur_err

  confusion:zero()
  epoch = epoch + 1
end

function test()
  test_count = test_count or 1
  local f = 0
  local time = sys.clock()
  model:evaluate()
  print ('==> testing on test set:')
  for t = 1, nTesting do
    local input = testset_input[{{t, {}}}]
    input = torch.reshape(input, input:size(1), input:size(2), 1):cuda()
    local target = testset_target[t][1]+1
    local pred = model:forward(input)
    local err = criterion:forward(pred, target)
    f = f + err
    confusion:add(pred, target)
  end

  time = sys.clock() - time
  time = time / nTesting
  print("==> time to test 1 sample = " .. (time*1000) .. 'ms')

  print(confusion)
  -- test_score = confusion.totalValid
  f = f/nTesting

  result_test_accu[test_count] = confusion.totalValid
  result_test_err[test_count] = f

  confusion:zero()
  test_count = test_count + 1
end

Maxiter = 100
iter = 0
while iter < Maxiter do
  train()
  test()
  iter = iter + 1
end

if option.pretraining then
  recordfile = hdf5.open('result_0721_pre.h5', 'w')
else
  recordfile = hdf5.open('result_0721_wo_pre.h5', 'w')
end
recordfile:write('/train_accu', result_train_accu)
recordfile:write('/train_err', result_train_err)
recordfile:write('/test_accu', result_test_accu)
recordfile:write('/test_err', result_test_err)
recordfile:close()
-- table.save(result_train, 'result_train_pre')
-- table.save(result_test, 'result_test_pre')
require 'gnuplot'
gnuplot.plot({'Train', result_train_err}, {'Test', result_test_err})
