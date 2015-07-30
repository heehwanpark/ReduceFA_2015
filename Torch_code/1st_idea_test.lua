require 'hdf5'
require 'cutorch'
require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
print '==> load datasets'

local datafile = hdf5.open('/home/heehwan/Documents/workspace/data/ReduceFA/data_normed_20sec_ECGII.h5', 'r')
local pretrainset = datafile:read('/pretrain'):all()
local labelset_input = datafile:read('/input'):all()
local labelset_target = datafile:read('/target'):all()
datafile:close()

pretrainset = pretrainset:transpose(1,2)
labelset_input = labelset_input:transpose(1,2)
labelset_target = labelset_target:transpose(1,2)

-- function normalize(data)
--   N = data:size(1)
--   for i = 1,N do
--     row = data[{i,{}}]
--     mean = torch.mean(row)
--     std = torch.std(row)
--     data[{i,{}}] = (row - mean)/std*1000
--   end
-- end
--
-- normalize(pretrainset)
-- normalize(labelset_input)

nFold = 10
nElement = labelset_target:size(1)
nTraining = nElement - math.floor(nElement/nFold)
nTesting = nElement - nTraining

nPretrain = pretrainset:size(1)

trainset_input = labelset_input[{{1,nTraining}, {}}]
trainset_target = labelset_target[{{1,nTraining}, {}}]

testset_input = labelset_input[{{nTraining+1, nElement}, {}}]
testset_target = labelset_target[{{nTraining+1, nElement}, {}}]

----------------------------------------------------------------------
print '==> define parameters'

-- Label: normal = 0, Asystole = 1, Bradycardia = 2, Tachycardia = 3,
-- Ventricular Tachycardia = 4, Ventricular Flutter/Fibrillation = 5
noutputs = 2

nfeatures = 1
width = 5000
height = 1
ninputs = nfeatures*width*height

nstates = {20, 50}
filter_size = 10
pool_size = 4

---------------------------------------------------------------------
print '==> construct model'

model = nn.Sequential()

-- 1st convolution layer
model:add(nn.Reshape(nfeatures, width, 1))
model:add(nn.SpatialConvolutionMM(1, nstates[1], 1, filter_size))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(1, pool_size))

-- Calculate # of outputs
nConvOut = math.floor((ninputs - filter_size + 1)/pool_size)

-- 2nd convolution layer
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], 1, filter_size))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(1, pool_size))

-- Calculate # of outputs
nConvOut = math.floor((nConvOut - filter_size + 1)/pool_size)

-- Standard MLP
model:add(nn.View(nstates[2]*nConvOut*1))
model:add(nn.Linear(nstates[2]*nConvOut*1, 1024))
model:add(nn.ReLU())

model:add(nn.Linear(1024, 1024))
model:add(nn.ReLU())
-- model:add(nn.Dropout(0.25))
model:add(nn.Linear(1024, noutputs))
model:add(nn.LogSoftMax())

model:cuda()

print(model)

----------------------------------------------------------------------
print '==> define loss'

weight = torch.Tensor(2)
weight[1] = 0.38
weight[2] = 0.62
criterion = nn.ClassNLLCriterion(weight)

-- criterion = nn.ClassNLLCriterion()
criterion:cuda()

----------------------------------------------------------------------
require 'optim'

print '==> defining some tools'

classes = {'Normal','Abnormal'}

-- require 'ConfusionMatrix_HH'
confusion = optim.ConfusionMatrix(classes)

parameters, gradParameters = model:getParameters()

----------------------------------------------------------------------
print '==> configuring optimizer'
optimState = {
  learningRate = 0.01,
  weightDecay = 0,
  momentum = 0,
  learningRateDecay = 0
}
optimMethod = optim.sgd

----------------------------------------------------------------------
print '==> defining training procedure'

batchsize = 5

function train()
  epoch = epoch or 1

  local time = sys.clock()

  model:training()

  shuffle = torch.randperm(nTraining)

  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchsize .. ']')

  cur_err = 0
  for t = 1, math.floor(nTraining/batchsize) do
    local inputs = {}
    local targets = {}
    batchstart = (t-1)*batchsize+1
    for i = batchstart,batchstart+batchsize-1 do
      local input = trainset_input[{{shuffle[i], {}}}]:cuda()
      -- input = torch.reshape(input, input:size(1), input:size(2), 1):cuda()
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
        -- print(output)
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
    -- optimMethod(feval, parameters, optimState)
    x,fx = optimMethod(feval, parameters, optimState)
    cur_err = cur_err + fx[1]
  end
  cur_err = cur_err/math.floor(nTraining/batchsize)
  print(cur_err)
  time = sys.clock() - time
  time = time / nTraining

  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

  print(confusion)

  confusion:zero()
  epoch = epoch + 1
end

function test()
  local time = sys.clock()
  model:evaluate()
  print ('==> testing on test set:')
  for t = 1, nTesting do
    local input = testset_input[{{t, {}}}]:cuda()
    -- input = torch.reshape(input, input:size(1), input:size(2), 1):cuda()
    local target = testset_target[t][1]+1
    local pred = model:forward(input)
    confusion:add(pred, target)
  end

  time = sys.clock() - time
  time = time / nTesting
  print("==> time to test 1 sample = " .. (time*1000) .. 'ms')
  print(confusion)

  confusion:zero()
end

Maxiter = 100
iter = 0
while iter < Maxiter do
  train()
  test()
  iter = iter + 1
end
