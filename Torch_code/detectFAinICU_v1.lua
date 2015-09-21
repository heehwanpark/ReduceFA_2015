require 'torch'
require 'custom_HH'
require 'optim'

function detectFAinICU_v1(convlayer_num, pool_size, testnum)
  print '==> Processing options'

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Reducing False Alarms using CNNs')
  cmd:text()
  cmd:text('Options:')
  -- Model
  -- Label: normal = 0, Asystole = 1, Bradycardia = 2, Tachycardia = 3,
  -- Ventricular Tachycardia = 4, Ventricular Flutter/Fibrillation = 5
  cmd:option('-nTarget', 2)
  cmd:option('-nInputFeature', 1)
  cmd:option('-inputSize', 2500) -- 250Hz * 10sec
  --- For convolutional networks
  cmd:option('-convlayer_num', convlayer_num)
  cmd:option('-nFeatures_c', 75)
  -- cmd:option('-nFeatures_c5', 60)
  -- cmd:option('-nFeatures_c6', 60)
  -- cmd:option('-nFeatures_c7', 60)
  --- For MLP
  cmd:option('-nFeatures_m1', 500)
  -- --- For PSD
  -- cmd:option('-lambda', 1)
  -- cmd:option('-beta', 1)
  -- Experiment Setting
  cmd:option('-seed', 1)
  cmd:option('-batchsize', 5)
  cmd:option('-nFold', 5)
  cmd:option('-maxIter', 200)
  cmd:option('-lr_sup', 0.001, 'Learning rate')
  cmd:option('-lr_unsup', 5e-6, 'Learning rate')
  cmd:option('-lrdecay',1e-5, 'Learning rate decay')
  cmd:option('-momentum', 0)
  cmd:option('-pretraining', false)
  -- Conv Setting
  cmd:option('-kernel', 25)
  cmd:option('-pool', pool_size)
  -- Torch Setting
  cmd:option('-thread', 16)
  -- File name
  cmd:option('-filename', '0917_' .. testnum)

  cmd:text()
  option = cmd:parse(arg)
  ----------------------------------------------------------------------
  print '==> Setting'

  torch.manualSeed(option.seed)
  torch.setnumthreads(option.thread)
  ----------------------------------------------------------------------
  require 'cutorch'
  require 'cunn'

  print '==> Construct CNN model'
  print '==> construct model'

  model = nn.Sequential()

  for i = 1, option.convlayer_num do
    if i == 1 then
      -- 1st convolution layer
      model:add(nn.SpatialConvolutionMM(option.nInputFeature, option.nFeatures_c, 1, option.kernel))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(1, option.pool))
      -- Calculate # of outputs
      nConvOut = math.floor((option.inputSize - option.kernel + 1)/option.pool)
    else
      -- 2nd+ convolution layer
      model:add(nn.SpatialConvolutionMM(option.nFeatures_c, option.nFeatures_c, 1, option.kernel))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(1, option.pool))
      -- Calculate # of outputs
      nConvOut = math.floor((nConvOut - option.kernel + 1)/option.pool)
    end
  end
  -- Standard MLP
  model:add(nn.Reshape(option.nFeatures_c*nConvOut*1))

  model:add(nn.Linear(option.nFeatures_c*nConvOut*1, option.nFeatures_m1))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))

  model:add(nn.Linear(option.nFeatures_m1, option.nFeatures_m1))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))

  -- model:add(nn.Linear(option.nFeatures_m1, option.nFeatures_m1))
  -- model:add(nn.ReLU())
  -- model:add(nn.Dropout(0.5))

  model:add(nn.Linear(option.nFeatures_m1, option.nTarget))
  model:add(nn.LogSoftMax())

  model:cuda()
  print(model)
  ----------------------------------------------------------------------
  print '==> Defining loss'

  -- weight = torch.Tensor(2)
  -- weight[1] = 1
  -- weight[2] = 5
  -- criterion = nn.ClassNLLCriterion(weight)
  criterion = nn.ClassNLLCriterion()
  criterion:cuda()
  ----------------------------------------------------------------------
  print '==> Defining some tools'

  classes = {'False Alarm','True Alarm'}
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

  result_train_accu = torch.zeros(option.maxIter)
  result_train_err = torch.zeros(option.maxIter)

  result_test_accu = torch.zeros(option.maxIter)
  result_test_err = torch.zeros(option.maxIter)
  result_test_conf = torch.zeros(option.maxIter, 4)

  Maxiter = option.maxIter
  iter = 1

  require 'training_cnn'
  require 'testing_cnn'

  while iter <= Maxiter do
    train()
    test()
    iter = iter + 1
  end

  folder = '/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/'
  if option.pretraining then
    recordfile = hdf5.open(folder .. option.filename .. '_wi_pre.h5', 'w')
  else
    recordfile = hdf5.open(folder .. option.filename .. '_wo_pre.h5', 'w')
  end
  recordfile:write('/train_accu', result_train_accu)
  recordfile:write('/train_err', result_train_err)
  recordfile:write('/test_accu', result_test_accu)
  recordfile:write('/test_err', result_test_err)
  recordfile:write('/test_confmatrix', result_test_conf)
  recordfile:close()
  -- table.save(result_train, 'result_train_pre')
  -- table.save(result_test, 'result_test_pre')
  -- require 'gnuplot'
  -- gnuplot.plot({'Train', result_train_err}, {'Test', result_test_err})
end
