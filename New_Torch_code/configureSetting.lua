require 'optim'

function configureSetting(option)
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
    learningRate = option.lr,
    weightDecay = 0,
    momentum = option.momentum,
    learningRateDecay = option.lr_decay
  }
  optimMethod = optim.sgd
  ----------------------------------------------------------------------
  print '==> Defining training procedure'

  parameters, gradParameters = model:getParameters()
  batchsize = option.batchsize
  max_iter = option.max_iter
  -- max_upIter = option.max_upIter
  -- max_iter = math.floor(max_upIter/math.floor(nTraining/batchsize))
end
