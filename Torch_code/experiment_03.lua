require 'torch'
require 'optim'

function experiment_02(ex_type, data_type)
  print '==> Processing options'

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Reducing False Alarms using CNNs')
  cmd:text()
  cmd:text('Options:')
  -- Data: 'chal600', 'mimic600', 'chal300+mimic300', 'chal600+mimic600', 'chal600+mimic1200', 'chal600+mimic2400', 'chal600+mimicAll'
  cmd:option('-datatype', data_type)
  -- Model
  cmd:option('-nTarget', 2)
  cmd:option('-nInputFeature', 1)
  cmd:option('-inputSize', 2500) -- 250Hz * 10sec
  --- For convolutional networks
  cmd:option('-convlayer_num', 3)
  cmd:option('-nFeatures_c', 75)
  --- For MLP
  cmd:option('-mlplayer_num', 1)
  cmd:option('-nUnit_mlp', 500)
  -- Experiment Setting
  cmd:option('-dbseed', 1)
  cmd:option('-weightseed', 1)
  cmd:option('-batchsize', 30)
  cmd:option('-nFold', 5)
  cmd:option('-max_upIter', 42000) -- Update iteration, Not epoch
  cmd:option('-lr_sup', 0.001, 'Learning rate')
  cmd:option('-lrdecay',1e-5, 'Learning rate decay')
  cmd:option('-momentum', 0)
  cmd:option('-dropout_rate', 0.5)
  cmd:option('-pretraining', false)
  -- Conv Setting
  cmd:option('-kernel', 250)
  cmd:option('-pool', 2)
  -- Torch Setting
  cmd:option('-thread', 16)
  -- File name
  cmd:option('-foldername', '/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/')
  cmd:option('-filename', 'experiment_02/'.. ex_type .. '/' .. data_type)

  cmd:text()
  option = cmd:parse(arg or {})

  torch.setnumthreads(option.thread)
  ----------------------------------------------------------------------
  require 'getLearningData'
  require 'getLearningData_testmix'
  require 'getLearningData_newidea'

  -- nTraining, trainset_input, trainset_target, nTesting, testset_input, testset_target = getLearningData(option)
  nTraining, trainset_input, trainset_target, nTesting, testset_input, testset_target = getLearningData_testmix(option)
  -- nTraining, trainset_input, trainset_target, nTesting, testset_input, testset_target = getLearningData_newidea(option)


  print(trainset_input:size())
  print(testset_input:size())
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

  -- train_accu = torch.zeros(option.max_upIter, 1)
  -- train_err = torch.zeros(option.max_upIter, 1)

  test_accu = torch.zeros(option.max_upIter, 1)
  test_err = torch.zeros(option.max_upIter, 1)
  test_conf = torch.zeros(option.max_upIter, 4)

  Conv_weight1 = torch.zeros(2, option.nFeatures_c, option.kernel)
  Conv_weight2 = torch.zeros(2, option.nFeatures_c, option.nFeatures_c*option.kernel)
  Conv_weight3 = torch.zeros(2, option.nFeatures_c, option.nFeatures_c*option.kernel)

  m1 = model.modules[1].weight:float()
  m2 = model.modules[4].weight:float()
  m3 = model.modules[7].weight:float()

  Conv_weight1[{1,{},{}}] = m1
  Conv_weight2[{1,{},{}}] = m2
  Conv_weight3[{1,{},{}}] = m3
  ----------------------------------------------------------------------
  require 'training_cnn_02'

  print '==> Start training'

  iterPerEpoch = math.floor(nTraining/batchsize)
  Maxiter = math.floor(option.max_upIter/iterPerEpoch)
  print('==> # of max iteration: ' .. Maxiter)
  iter = 1
  upIter = 1
  while iter <= Maxiter do
    train()

    if iter == Maxiter then
      m1 = model.modules[1].weight:float()
      m2 = model.modules[4].weight:float()
      m3 = model.modules[7].weight:float()

      Conv_weight1[{2,{},{}}] = m1
      Conv_weight2[{2,{},{}}] = m2
      Conv_weight3[{2,{},{}}] = m3
    end
    iter = iter + 1
  end
  ----------------------------------------------------------------------
  recordfile = hdf5.open(option.foldername .. option.filename .. '.h5', 'w')
  -- recordfile:write('/train_accu', train_accu)
  -- recordfile:write('/train_err', train_err)
  recordfile:write('/test_accu', test_accu)
  recordfile:write('/test_err', test_err)
  recordfile:write('/test_confmatrix', test_conf)
  recordfile:write('/conv1_weight', Conv_weight1)
  recordfile:write('/conv2_weight', Conv_weight2)
  recordfile:write('/conv3_weight', Conv_weight3)
  recordfile:close()
end
