require 'torch'
require 'optim'

function experiment_01(data_type, db_seed, weight_seed)
  print '==> Processing options'

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Reducing False Alarms using CNNs')
  cmd:text()
  cmd:text('Options:')
  -- Data: 'chal', 'mimic+chal_all', 'mimic+chal_small'
  cmd:option('-datatype', data_type)
  -- Model
  cmd:option('-nTarget', 2)
  cmd:option('-nInputFeature', 1)
  cmd:option('-inputSize', 2500) -- 250Hz * 10sec
  --- For convolutional networks
  cmd:option('-convlayer_num', 3)
  cmd:option('-nFeatures_c', 75)
  --- For MLP
  cmd:option('-mlplayer_num', 2)
  cmd:option('-nUnit_mlp', 500)
  -- Experiment Setting
  cmd:option('-dbseed', db_seed)
  cmd:option('-weightseed', weight_seed)
  cmd:option('-batchsize', 10)
  cmd:option('-nFold', 5)
  cmd:option('-maxIter', 300)
  cmd:option('-lr_sup', 0.001, 'Learning rate')
  cmd:option('-lrdecay',1e-5, 'Learning rate decay')
  cmd:option('-momentum', 0)
  cmd:option('-dropout_rate', 0.5)
  cmd:option('-pretraining', false)
  -- Conv Setting
  cmd:option('-kernel', 25)
  cmd:option('-pool', 5)
  -- Torch Setting
  cmd:option('-thread', 16)
  -- File name
  cmd:option('-foldername', '/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/')
  cmd:option('-filename', 'experiment_01/' .. data_type .. '_init_' .. weight_seed .. '_batch_' .. 10)

  cmd:text()
  option = cmd:parse(arg)

  torch.setnumthreads(option.thread)
  ----------------------------------------------------------------------
  require 'getLearningData'

  nTraining, trainset_input, trainset_target, nTesting, testset_input, testset_target = getLearningData(option)
  ----------------------------------------------------------------------
  require 'buildCNNModel'

  model = buildCNNModel(option)
  ----------------------------------------------------------------------
  print '==> Defining loss'

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

  parameters, gradParameters = model:getParameters()
  batchsize = option.batchsize

  train_accu = torch.zeros(option.maxIter, 1)
  train_err = torch.zeros(option.maxIter, 1)

  test_accu = torch.zeros(option.maxIter, 1)
  test_err = torch.zeros(option.maxIter, 1)
  test_conf = torch.zeros(option.maxIter, 4)

  Maxiter = option.maxIter
  iter = 1

  ----------------------------------------------------------------------
  print '==> Start training'

  require 'training_cnn'
  require 'testing_cnn'

  while iter <= Maxiter do
    train()
    test()
    iter = iter + 1
  end
  ----------------------------------------------------------------------
  recordfile = hdf5.open(option.foldername .. option.filename .. '.h5', 'w')
  recordfile:write('/train_accu', train_accu)
  recordfile:write('/train_err', train_err)
  recordfile:write('/test_accu', test_accu)
  recordfile:write('/test_err', test_err)
  recordfile:write('/test_confmatrix', test_conf)
  recordfile:close()
end
