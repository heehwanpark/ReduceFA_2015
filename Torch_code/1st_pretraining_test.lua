require 'hdf5'
require 'unsup'
require 'optim'

-- torch.setdefaulttensortype('torch.FloatTensor')

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
--     mean = torch.mean(data[{i,{}}])
--     std = torch.std(data[{i,{}}])
--     data[{i,{}}] = (data[{i,{}}] - mean)/std
--   end
-- end

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
noutputs = 6

nfeatures = 1
width = 5000
height = 1
ninputs = nfeatures*width*height

nstates = {16, 32}
filter_size = 10
pool_size = 4


----------------------------------------------------------------------
print '==> construct Pre-training model'

-- 1st Convolusion layer

-- connection table:
conntable_1 = nn.tables.full(1, nstates[1])

-- decoder's table:
local decodertable_1 = conntable_1:clone()
decodertable_1[{ {},1 }] = conntable_1[{ {},2 }]
decodertable_1[{ {},2 }] = conntable_1[{ {},1 }]
local outputSize = conntable_1[{ {},2 }]:max()

encoder = nn.Sequential()
encoder:add(nn.SpatialConvolutionMap(conntable_1, 1, filter_size))
encoder:add(nn.ReLU())
encoder:add(nn.Diag(outputSize))

decoder = unsup.SpatialConvFistaL1(decodertable_1, 1, filter_size, 1, 5000, 1)

-- module
pretraining_module = unsup.PSD(encoder, decoder, 1)

----------------------------------------------------------------------
print '==> defining Pre-training procedure'

batchsize = 10

x, dl_dx = pretraining_module:getParameters()

sgdconf = {learningRate = 1e-5}

function pretrain()

  epoch = epoch or 1

  local time = sys.clock()

  print('==> doing epoch on pre-training data:')
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchsize .. ']')
  cur_err = 0;
  for t = 1, math.floor(nPretrain/batchsize) do
    samples = {}

    batchstart = (t-1)*batchsize+1
    for i = batchstart,batchstart+batchsize-1 do
      local input = pretrainset[{i, {}}]:clone()
      input = torch.reshape(input, 1, input:size(1), 1)
      table.insert(samples, input)
    end

    local feval = function()

      local f = 0
      dl_dx:zero()

      for i = 1, batchsize do
        f = f + pretraining_module:updateOutput(samples[i], samples[i])

        pretraining_module:updateGradInput(samples[i], samples[i])
        pretraining_module:accGradParameters(samples[i], samples[i])
      end

      dl_dx:div(batchsize)
      f = f/batchsize
      print(f)
      return f, dl_dx
    end

    -- optim.sgd(feval, x, sgdconf)
    _x,_fx = optim.sgd(feval, x, sgdconf)
    cur_err = cur_err + _fx[1]

    pretraining_module:normalize()
  end

  epoch = epoch + 1
end

Maxiter = 100
iter = 0
while iter < Maxiter do
  pretrain()
  iter = iter + 1
end
