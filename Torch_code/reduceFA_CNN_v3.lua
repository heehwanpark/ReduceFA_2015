require 'torch'
require 'custom_HH'
require 'optim'

----------------------------------------------------------------------
print '==> Processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Reducing False Alarms using CNNs')
cmd:text()
cmd:text('Options:')
-- Data
cmd:option('-chaldata', '/home/salab/Documents/workspace/data/ReduceFA/chal2015_data_10sec_resampled.h5')
cmd:option('-mitdata', '/home/salab/Documents/workspace/data/ReduceFA/mitbih_data_10sec_v2.h5')
cmd:option('-datatype', 'chal-2015') -- chal-2015, mitbih, mit+chal
-- Model
-- Label: normal = 0, Asystole = 1, Bradycardia = 2, Tachycardia = 3,
-- Ventricular Tachycardia = 4, Ventricular Flutter/Fibrillation = 5
cmd:option('-nTarget', 2)
cmd:option('-nInputFeature', 1)
cmd:option('-inputSize', 3600) -- 360Hz * 10sec
--- For convolutional networks
cmd:option('-nFeatures_c1', 60)
cmd:option('-nFeatures_c2', 60)
cmd:option('-nFeatures_c3', 60)
cmd:option('-nFeatures_c4', 60)
-- cmd:option('-nFeatures_c5', 60)
-- cmd:option('-nFeatures_c6', 60)
-- cmd:option('-nFeatures_c7', 60)
--- For MLP
cmd:option('-nFeatures_m1', 500)
--- For PSD
cmd:option('-lambda', 1)
cmd:option('-beta', 1)
-- Experiment Setting
cmd:option('-seed', 1)
cmd:option('-batchsize', 5)
cmd:option('-nFold', 5)
cmd:option('-maxIter', 200)
cmd:option('-lr_sup', 0.005, 'Learning rate')
cmd:option('-lr_unsup', 5e-6, 'Learning rate')
cmd:option('-lrdecay',1e-5, 'Learning rate decay')
cmd:option('-momentum', 0)
cmd:option('-pretraining', false)
-- Conv Setting
cmd:option('-kernel', 10)
cmd:option('-pool', 4)
-- Torch Setting
cmd:option('-thread', 16)
-- File name
cmd:option('-filename', '0806_c4_p4')

cmd:text()
option = cmd:parse(arg)
----------------------------------------------------------------------
print '==> Setting'

torch.manualSeed(option.seed)
torch.setnumthreads(option.thread)
----------------------------------------------------------------------
print '==> Load datasets'

require 'hdf5'
if option.datatype == 'mitbih'  then
  mit_datafile = hdf5.open(option.mitdata, 'r')
  mit_labelset_input = mit_datafile:read('/inputs'):all()
  mit_labelset_target = mit_datafile:read('/targets'):all()
  mit_datafile:close()
elseif option.datatype == 'chal-2015' then
  chal_datafile = hdf5.open(option.chaldata, 'r')
  chal_pretrainset = chal_datafile:read('/pretrain'):all()
  chal_labelset_input = chal_datafile:read('/input'):all()
  chal_labelset_target = chal_datafile:read('/target'):all()
  chal_datafile:close()

  chal_pretrainset = chal_pretrainset:transpose(1,2)
  chal_labelset_input = chal_labelset_input:transpose(1,2)
  chal_labelset_target = chal_labelset_target:transpose(1,2)
elseif option.datatype == 'mit+chal' then
  mit_datafile = hdf5.open(option.mitdata, 'r')
  mit_labelset_input = mit_datafile:read('/inputs'):all()
  mit_labelset_target = mit_datafile:read('/targets'):all()
  mit_datafile:close()

  chal_datafile = hdf5.open(option.chaldata, 'r')
  chal_pretrainset = chal_datafile:read('/pretrain'):all()
  chal_labelset_input = chal_datafile:read('/input'):all()
  chal_labelset_target = chal_datafile:read('/target'):all()
  chal_datafile:close()

  chal_pretrainset = chal_pretrainset:transpose(1,2)
  chal_labelset_input = chal_labelset_input:transpose(1,2)
  chal_labelset_target = chal_labelset_target:transpose(1,2)
else
  error('==> datatype error!!')
end
----------------------------------------------------------------------
if option.pretraining then
  print '==> Set pretraining'
  require 'unsup'
  require 'ConvPSD_HH'

  print '...Pre-training 1st layer'
  -- 1st layer
  pretrainset1 = convertForPretrain(chal_pretrainset, option.inputSize)
  -- encoder1, decoder1 = trainConvPSD(pretrainset1, option.nInputFeature, option.nFeatures_c1, option, 'pretrain_result_layer1')
  encoder1 = torch.load('pretrain_result_layer1_encoder.net')
  decoder1 = torch.load('pretrain_result_layer1_decoder.net')

  model_1 = nn.Sequential()
  model_1:add(nn.SpatialConvolutionMM(option.nInputFeature, option.nFeatures_c1, 1, option.kernel))
  model_1:add(nn.ReLU())
  model_1:add(nn.SpatialMaxPooling(1, option.pool))
  model_1.modules[1].weight = encoder1.modules[1].weight

  print '...Pre-training 2nd layer'
  -- 2nd layer
  pretrainset2 = netsThrough(model_1, pretrainset1)
  encoder2, decoder2 = trainConvPSD(pretrainset2, option.nFeatures1, option.nFeatures2, option, 'pretrain_result_layer2')
end
----------------------------------------------------------------------
require 'cutorch'
require 'cunn'

print '==> Construct CNN model'
print '==> construct model'

model = nn.Sequential()

-- 1st convolution layer
model:add(nn.SpatialConvolutionMM(option.nInputFeature, option.nFeatures_c1, 1, option.kernel))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(1, option.pool))

-- Calculate # of outputs
nConvOut = math.floor((option.inputSize - option.kernel + 1)/option.pool)

-- 2nd convolution layer
model:add(nn.SpatialConvolutionMM(option.nFeatures_c1, option.nFeatures_c2, 1, option.kernel))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(1, option.pool))

-- Calculate # of outputs
nConvOut = math.floor((nConvOut - option.kernel + 1)/option.pool)

-- 3rd convolution layer
model:add(nn.SpatialConvolutionMM(option.nFeatures_c2, option.nFeatures_c3, 1, option.kernel))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(1, option.pool))

-- Calculate # of outputs
nConvOut = math.floor((nConvOut - option.kernel + 1)/option.pool)

-- 4th convolution layer
model:add(nn.SpatialConvolutionMM(option.nFeatures_c3, option.nFeatures_c4, 1, option.kernel))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(1, option.pool))

-- Calculate # of outputs
nConvOut = math.floor((nConvOut - option.kernel + 1)/option.pool)

-- -- 5th convolution layer
-- model:add(nn.SpatialConvolutionMM(option.nFeatures_c4, option.nFeatures_c5, 1, option.kernel))
-- model:add(nn.ReLU())
-- model:add(nn.SpatialMaxPooling(1, option.pool))
--
-- Calculate # of outputs
-- nConvOut = math.floor((nConvOut - option.kernel + 1)/option.pool)

-- -- 6th convolution layer
-- model:add(nn.SpatialConvolutionMM(option.nFeatures_c5, option.nFeatures_c6, 1, option.kernel))
-- model:add(nn.ReLU())
-- model:add(nn.SpatialMaxPooling(1, option.pool))
--
-- -- Calculate # of outputs
-- nConvOut = math.floor((nConvOut - option.kernel + 1)/option.pool)

-- -- 7th convolution layer
-- model:add(nn.SpatialConvolutionMM(option.nFeatures_c6, option.nFeatures_c7, 1, option.kernel))
-- model:add(nn.ReLU())
-- model:add(nn.SpatialMaxPooling(1, option.pool))
--
-- -- Calculate # of outputs
-- nConvOut = math.floor((nConvOut - option.kernel + 1)/option.pool)

-- Standard MLP
model:add(nn.View(option.nFeatures_c4*nConvOut*1))
model:add(nn.Linear(option.nFeatures_c4*nConvOut*1, option.nFeatures_m1))
model:add(nn.ReLU())
model:add(nn.Linear(option.nFeatures_m1, option.nFeatures_m1))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(option.nFeatures_m1, option.nTarget))
model:add(nn.LogSoftMax())

if option.pretraining then
  model.modules[1].weight = encoder1.modules[1].weight
  -- model[4].weight = encoder2.modules[1].weight
end

model:cuda()
print(model)
----------------------------------------------------------------------
print '==> Defining loss'

-- weight = torch.Tensor(2)
-- weight[1] = 0.4
-- weight[2] = 0.6
-- criterion = nn.ClassNLLCriterion(weight)
criterion = nn.ClassNLLCriterion()
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
  momentum = option.momentum,
  learningRateDecay = option.lrdecay
}
optimMethod = optim.sgd
----------------------------------------------------------------------
print '==> Defining training procedure'
-- reset randseed
torch.manualSeed(1)

parameters, gradParameters = model:getParameters()
batchsize = option.batchsize
nFold = option.nFold

if option.datatype == 'mitbih' then
-- Training & Testing: MIT-BIH
  nElement = mit_labelset_target:size(1)
  nTraining = nElement - math.floor(nElement/nFold)
  nTesting = nElement - nTraining

  shuffle = torch.randperm(nElement)

  trainset_input = torch.zeros(nTraining, option.inputSize)
  trainset_target = torch.zeros(nTraining, 1)
  for i = 1, nTraining do
    trainset_input[{i, {}}] = mit_labelset_input[{shuffle[i], {}}]
    trainset_target[i] = mit_labelset_target[shuffle[i]]
  end

  testset_input = torch.zeros(nTesting, option.inputSize)
  testset_target = torch.zeros(nTesting, 1)
  for j = 1, nTesting do
    testset_input[{j, {}}] = mit_labelset_input[{shuffle[j+nTraining], {}}]
    testset_target[j] = mit_labelset_target[shuffle[j+nTraining]]
  end

elseif option.datatype == 'mit+chal' then
-- Training: MIT-BIH + Chal-2015 pretraining, Testing: Chal-2015 last 10sec
  nMitSamples = mit_labelset_target:size(1)
  nChalSamples = chal_labelset_target:size(1)
  nChalTrain = nChalSamples - math.floor(nChalSamples/nFold)
  nChalTest = nChalSamples - nChalTrain

  Chal_shuffle = torch.randperm(nChalSamples)

  chal_trainset_input = torch.zeros(nChalTrain, option.inputSize)
  chal_trainset_target = torch.zeros(nChalTrain, 1)
  for i = 1, nChalTrain do
    chal_trainset_input[{i, {}}] = chal_labelset_input[{Chal_shuffle[i], {}}]
    chal_trainset_target[i] = chal_labelset_target[Chal_shuffle[i]]
  end

  chal_testset_input = torch.zeros(nChalTest, option.inputSize)
  chal_testset_target = torch.zeros(nChalTest, 1)
  for j = 1, nChalTest do
    chal_testset_input[{j, {}}] = chal_labelset_input[{Chal_shuffle[j+nChalTrain], {}}]
    chal_testset_target[j] = chal_labelset_target[Chal_shuffle[j+nChalTrain]]
  end

  nTraining = nMitSamples + nChalTrain
  nTesting = nChalTest
  nElement = nTraining + nTesting

  trainset_input = torch.zeros(nTraining, option.inputSize)
  trainset_target = torch.zeros(nTraining, 1)

  for i = 1, nMitSamples do
    trainset_input[{i, {}}] = mit_labelset_input[{i, {}}]
    trainset_target[i] = mit_labelset_target[i]
  end

  for i = 1, nChalTrain do
    trainset_input[{nMitSamples+i, {}}] = chal_trainset_input[{i, {}}]
    trainset_target[nMitSamples+i] = chal_trainset_target[i]
  end

  testset_input = chal_testset_input
  testset_target = chal_testset_target

elseif option.datatype == 'chal-2015' then
  -- Training & Testing: Chal-2015 last 10sec
  nChalSamples = chal_labelset_target:size(1)
  nChalTrain = nChalSamples - math.floor(nChalSamples/nFold)
  nChalTest = nChalSamples - nChalTrain

  Chal_shuffle = torch.randperm(nChalSamples)

  chal_trainset_input = torch.zeros(nChalTrain, option.inputSize)
  chal_trainset_target = torch.zeros(nChalTrain, 1)
  for i = 1, nChalTrain do
    chal_trainset_input[{i, {}}] = chal_labelset_input[{Chal_shuffle[i], {}}]
    chal_trainset_target[i] = chal_labelset_target[Chal_shuffle[i]]
  end

  chal_testset_input = torch.zeros(nChalTest, option.inputSize)
  chal_testset_target = torch.zeros(nChalTest, 1)
  for j = 1, nChalTest do
    chal_testset_input[{j, {}}] = chal_labelset_input[{Chal_shuffle[j+nChalTrain], {}}]
    chal_testset_target[j] = chal_labelset_target[Chal_shuffle[j+nChalTrain]]
  end

  nTraining = nChalTrain
  nTesting = nChalTest
  nElement = nChalSamples

  trainset_input = chal_trainset_input
  trainset_target = chal_trainset_target

  testset_input = chal_testset_input
  testset_target = chal_testset_target

else
  error('==> datatype error!!')
end

print(trainset_input:size())
print(testset_input:size())

result_train_accu = torch.zeros(option.maxIter)
result_train_err = torch.zeros(option.maxIter)

result_test_accu = torch.zeros(option.maxIter)
result_test_err = torch.zeros(option.maxIter)

function train()
  epoch = epoch or 1

  cur_err = 0

  local time = sys.clock()

  model:training()

  shuffle_t = torch.randperm(nTraining)

  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchsize .. ']')
  for t = 1, math.floor(nTraining/batchsize) do
    local inputs = {}
    local targets = {}
    batchstart = (t-1)*batchsize+1
    for i = batchstart,batchstart+batchsize-1 do
      local input = trainset_input[{{shuffle_t[i], {}}}]
      input = torch.reshape(input, input:size(1), input:size(2), 1):cuda()
      local target = trainset_target[shuffle_t[i]][1]+1
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

  local acc_list = torch.zeros(nTesting)

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

    y,i_y = torch.max(pred,1)
    if i_y[1] == target then
      acc_list[t] = 1
    end
  end

  -- if test_count == Maxiter then
  --   print(shuffle[{{nTraining+1, nElement}}]:size())
  --   print(acc_list:size())
  --   l = torch.cat(shuffle[{{nTraining+1, nElement}}], acc_list, 2)
  --   faultfile = hdf5.open('/home/heehwan/faultfile.h5', 'w')
  --   faultfile:write('/list', l)
  --   faultfile:close()
  -- end

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

Maxiter = option.maxIter
iter = 0
while iter < Maxiter do
  train()
  test()
  iter = iter + 1
end

folder = '/home/salab/Documents/workspace/data/ReduceFA/output/'
if option.pretraining then
  recordfile = hdf5.open(folder .. option.filename .. '_' .. option.datatype .. '_wi_pre.h5', 'w')
else
  recordfile = hdf5.open(folder .. option.filename .. '_' .. option.datatype ..'_wo_pre.h5', 'w')
end
recordfile:write('/train_accu', result_train_accu)
recordfile:write('/train_err', result_train_err)
recordfile:write('/test_accu', result_test_accu)
recordfile:write('/test_err', result_test_err)
recordfile:close()
-- table.save(result_train, 'result_train_pre')
-- table.save(result_test, 'result_test_pre')
-- require 'gnuplot'
-- gnuplot.plot({'Train', result_train_err}, {'Test', result_test_err})
