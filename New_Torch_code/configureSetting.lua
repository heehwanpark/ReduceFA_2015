require 'optim'

function configureSetting(option)
  print '==> Defining loss'

  -- if option.class_weight_switch then
  --   weight = torch.Tensor(2)
  --   weight[1] = option.class_weight[1]
  --   weight[2] = option.class_weight[2]
  --   criterion = nn.ClassNLLCriterion(weight)
  -- else
  --   criterion = nn.ClassNLLCriterion()
  -- end

  weight = torch.Tensor(4)
  weight[1] = 1
  weight[2] = 1
  weight[3] = 5
  weight[4] = 1

  require 'ClassNLLCriterion_HH'
  criterion = nn.ClassNLLCriterion_HH(weight)

  if option.cuda then
    criterion:cuda()
  end
  ----------------------------------------------------------------------
  print '==> Defining some tools'

  classes = {'False Alarm','True Alarm'}
  confusion = optim.ConfusionMatrix(classes)
  ----------------------------------------------------------------------
  print '==> configuring optimizer'

  optimState = {
    learningRate = option.lr,
    weightDecay = 0,
    momentum = option.momentum,
    learningRateDecay = option.lr_decay
  }

  require 'sgd_HH'
  optimMethod = optim.sgd_HH
  -- optimMethod = optim.sgd
  ----------------------------------------------------------------------
  print '==> Defining training procedure'

  parameters, gradParameters = model:getParameters()
  batchsize = option.batchsize
  max_iter = option.max_iter
  -- max_upIter = option.max_upIter
  -- max_iter = math.floor(max_upIter/math.floor(nTraining/batchsize))
end
