require 'torch'
require 'optim'
----------------------------------------------------------------------
function experiment_01_wavelet(db_seed)
  print '==> Processing options'

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Reducing False Alarms using CNNs')
  cmd:text()
  cmd:text('Options:')
  -- Data: 'chal', 'mimic+chal_all', 'mimic+chal_small'
  -- cmd:option('-datatype', data_type)
  -- Model
  cmd:option('-nTarget', 2)
  cmd:option('-nInputFeature', 4)
  cmd:option('-inputSize', 2500) -- 250Hz * 10sec
  --- For convolutional networks
  cmd:option('-convlayer_num', 3)
  cmd:option('-nFeatures_c', 75)
  --- For MLP
  cmd:option('-mlplayer_num', 1)
  cmd:option('-nUnit_mlp', 500)
  -- Experiment Setting
  cmd:option('-dbseed', db_seed)
  cmd:option('-weightseed', 1)
  cmd:option('-batchsize', 30)
  cmd:option('-nFold', 5)
  cmd:option('-maxIter', 300)
  cmd:option('-lr_sup', 0.001, 'Learning rate')
  cmd:option('-lrdecay',1e-5, 'Learning rate decay')
  cmd:option('-momentum', 0)
  cmd:option('-dropout_rate', 0.5)
  cmd:option('-pretraining', false)
  -- Conv Setting
  cmd:option('-kernel', 50)
  cmd:option('-pool', 5)
  -- Torch Setting
  cmd:option('-thread', 16)
  -- File name
  cmd:option('-foldername', '/home/heehwan/Workspace/Data/ReduceFA_2015/cnn_output/weirdmimic/')
  cmd:option('-filename', 'experiment_01/04_wavelet/mimic+chal_all_db_' .. db_seed .. '_init_' .. 1)

  cmd:text()
  option = cmd:parse(arg or {})

  torch.setnumthreads(option.thread)
  ----------------------------------------------------------------------
  require 'hdf5'

  print '==> Load datasets'

  local chal_file = hdf5.open('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5', 'r')
  local chal_input = chal_file:read('/wave_inputs'):all()
  local chal_target = chal_file:read('/targets'):all()
  chal_file:close()

  chal_input = chal_input:transpose(1,3)
  chal_target = chal_target:transpose(1,2)

  local mimic2_file = hdf5.open('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_v3.h5', 'r')
  local mimic2_input = mimic2_file:read('/wave_inputs'):all()
  local mimic2_target = mimic2_file:read('/targets'):all()
  mimic2_file:close()

  mimic2_input = mimic2_input:transpose(1,3)
  mimic2_target = mimic2_target:transpose(1,2)

  -- Fix random seed
  torch.manualSeed(option.dbseed)

  -- # of training elements in challenge 2015 set
  local chal_trainsize = 600
  local nEle_chal = chal_target:size(1)
  local nEle_mimic2 = mimic2_target:size(1)
  local shuffle = torch.randperm(nEle_chal)
  local inputSize = option.inputSize

  nTesting = nEle_chal - chal_trainsize
  testset_input = torch.zeros(nTesting, 4, inputSize)
  testset_target = torch.zeros(nTesting, 1)
  for i = 1, nTesting do
    testset_input[{i, {}, {}}] = chal_input[{shuffle[i], {}, {}}]
    testset_target[i] = chal_target[shuffle[i]]
  end

  nTraining = nEle_mimic2 + chal_trainsize
  trainset_input = torch.zeros(nTraining, 4, inputSize)
  trainset_target = torch.zeros(nTraining, 1)
  for i = 1, nTraining do
    if i <= nEle_mimic2 then
      trainset_input[{i, {}, {}}] = mimic2_input[{i, {}, {}}]
      trainset_target[i] = mimic2_target[i]
    else
      trainset_input[{i, {}, {}}] = chal_input[{shuffle[nTesting+i-(nEle_mimic2)], {}, {}}]
      trainset_target[i] = chal_target[shuffle[nTesting+i-(nEle_mimic2)]]
    end
  end
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

  -- Conv_weight1 = torch.zeros(option.maxIter, option.nFeatures_c, option.kernel)
  -- Conv_weight2 = torch.zeros(option.maxIter, option.nFeatures_c, option.nFeatures_c*option.kernel)

  Maxiter = option.maxIter
  ----------------------------------------------------------------------
  function train()
    local cur_err = 0
    local time = sys.clock()

    model:training()

    shuffle_t = torch.randperm(nTraining)

    print('\n==> doing epoch on training data:')
    print("==> online epoch # " .. iter .. ' [batchSize = ' .. batchsize .. ']')
    for t = 1, math.floor(nTraining/batchsize) do
      local inputs = {}
      local targets = {}
      batchstart = (t-1)*batchsize+1
      for i = batchstart, batchstart+batchsize-1 do
        local input = trainset_input[{{shuffle_t[i], {}, {}}}]
        input = input[1]
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

    print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    print(confusion)
    cur_err = cur_err/math.floor(nTraining/batchsize)
    print("==> current error = " .. cur_err)

    train_accu[iter] = confusion.totalValid
    train_err[iter] = cur_err

    confusion:zero()
  end
  ----------------------------------------------------------------------
  function test()
    local f = 0
    local time = sys.clock()

    model:evaluate()

    print ('\n==> testing on test set:')
    for t = 1, nTesting do
      local input = testset_input[{{t, {}, {}}}]
      input = input[1]
      input = torch.reshape(input, input:size(1), input:size(2), 1):cuda()
      local target = testset_target[t][1]+1
      local pred = model:forward(input)
      local err = criterion:forward(pred, target)
      f = f + err
      confusion:add(pred, target)
    end

    time = sys.clock() - time
    time = time / nTesting
    print("==> time to test 1 sample = " .. (time*1000) .. 'ms')

    print(confusion)
    -- test_score = confusion.totalValid
    f = f/nTesting

    test_accu[iter] = confusion.totalValid
    test_err[iter] = f

    cm = confusion.mat
    test_conf[iter][1] = cm[1][1]
    test_conf[iter][2] = cm[1][2]
    test_conf[iter][3] = cm[2][1]
    test_conf[iter][4] = cm[2][2]

    confusion:zero()
  end
  ----------------------------------------------------------------------
  print '==> Start training'

  iter = 1
  while iter <= Maxiter do
    train()
    test()

    -- m1 = model.modules[1].weight:float()
    -- m2 = model.modules[4].weight:float()
    --
    -- Conv_weight1[{iter,{},{}}] = m1
    -- Conv_weight2[{iter,{},{}}] = m2

    iter = iter + 1
  end
  ----------------------------------------------------------------------
  recordfile = hdf5.open(option.foldername .. option.filename .. '.h5', 'w')
  recordfile:write('/train_accu', train_accu)
  recordfile:write('/train_err', train_err)
  recordfile:write('/test_accu', test_accu)
  recordfile:write('/test_err', test_err)
  recordfile:write('/test_confmatrix', test_conf)
  -- recordfile:write('/conv1_weight', Conv_weight1)
  -- recordfile:write('/conv2_weight', Conv_weight2)
  recordfile:close()
end
