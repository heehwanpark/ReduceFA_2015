require 'torch'
require 'optim'

function experiment_03(mlp_architecture, data_type, maxmin_type)
  arch_str = '['
  for i = 1, table.getn(mlp_architecture) do
    if i == table.getn(mlp_architecture) then
      arch_str = arch_str .. mlp_architecture[i] .. ']'
    else
      arch_str = arch_str .. mlp_architecture[i] .. '-'
    end
  end

  print('==> Architecture' .. ': ' .. arch_str)

  print '==> Processing options'

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Reducing False Alarms using DNNs')
  cmd:text()
  cmd:text('Options:')
  -- Data: 'chal600+mimicAll'
  cmd:option('-datatype', data_type)
  -- Model
  cmd:option('-nTarget', 2)
  cmd:option('-nInputFeature', 1)
  cmd:option('-inputSize', 2500) -- 250Hz * 10sec
  --- For MLP
  cmd:option('-mlp_architecture', mlp_architecture)
  -- Experiment Setting
  cmd:option('-dbseed', 1)
  cmd:option('-weightseed', 1)
  cmd:option('-batchsize', 30)
  cmd:option('-maxmin_type', maxmin_type)
  cmd:option('-maxmin_window', 50)
  cmd:option('-nFold', 5)
  cmd:option('-maxIter', 200)
  cmd:option('-lr_sup', 0.001, 'Learning rate')
  cmd:option('-lrdecay', 1e-5, 'Learning rate decay')
  cmd:option('-momentum', 0)
  cmd:option('-dropout_rate', 0.5)
  -- Torch Setting
  cmd:option('-thread', 16)
  -- File name
  cmd:option('-foldername', '/home/heehwan/Workspace/Data/ReduceFA_2015/dnn_output/')
  cmd:option('-filename', arch_str .. '-' .. data_type .. '-' .. maxmin_type)

  cmd:text()
  option = cmd:parse(arg or {})

  torch.setnumthreads(option.thread)
  ----------------------------------------------------------------------
  require 'getLearningData_testmix'

  nTraining, trainset_input, trainset_target, nTesting, testset_input, testset_target = getLearningData_testmix(option)

  require 'custom_HH'
  if option.maxmin_type == 'max' then
    trainset_input = makeMax(trainset_input)
    testset_input = makeMax(testset_input)
    option.inputSize = option.inputSize - option.maxmin_window + 1
  elseif option.maxmin_type == 'min' then
    trainset_input = makeMin(trainset_input)
    testset_input = makeMin(testset_input)
    option.inputSize = option.inputSize - option.maxmin_window + 1
  end

  print(trainset_input:size())
  print(testset_input:size())
  ----------------------------------------------------------------------
  require 'buildDNNModel'

  model = buildDNNModel(option)
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
  Maxiter = option.maxIter

  Maxupiter = Maxiter*math.floor(nTraining/batchsize)

  test_accu = torch.zeros(Maxupiter, 1)
  test_err = torch.zeros(Maxupiter, 1)
  test_conf = torch.zeros(Maxupiter, 4)
  ----------------------------------------------------------------------
  require 'training_dnn'
  require 'testing_dnn'

  print '==> Start training'

  print('==> # of max iteration: ' .. Maxiter)
  iter = 1
  upIter = 1
  while iter <= Maxiter do
    train()
    -- test()
    iter = iter + 1
  end
  ----------------------------------------------------------------------
  recordfile = hdf5.open(option.foldername .. option.filename .. '.h5', 'w')
  -- recordfile:write('/train_accu', train_accu)
  -- recordfile:write('/train_err', train_err)
  recordfile:write('/test_accu', test_accu)
  recordfile:write('/test_err', test_err)
  recordfile:write('/test_confmatrix', test_conf)
  recordfile:close()
end
