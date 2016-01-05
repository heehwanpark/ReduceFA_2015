require 'optim'

function configureSetting(option)
  print '==> Defining loss'

  if option.class_weight_switch then
    if table.getn(option.class_weight) ~= 4 then
      error("Class weight should be an 1-by-4 table")
    else
      weight = torch.Tensor(4)
      weight[1] = option.class_weight[1]
      weight[2] = option.class_weight[2]
      weight[3] = option.class_weight[3]
      weight[4] = option.class_weight[4]
    end

    require 'ClassNLLCriterion_HH'
    criterion = nn.ClassNLLCriterion_HH(weight)
  else
    criterion = nn.ClassNLLCriterion()
  end

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

  if option.gaussian_noise then
    require 'sgd_HH'
    optimMethod = optim.sgd_HH
  else
    optimMethod = optim.sgd
  end
  ----------------------------------------------------------------------
  print '==> Defining training procedure'

  parameters, gradParameters = model:getParameters()
  batchsize = option.batchsize
  max_iter = option.max_iter
  -- max_upIter = option.max_upIter
  -- max_iter = math.floor(max_upIter/math.floor(nTraining/batchsize))
end
