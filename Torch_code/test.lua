function test()
  local f = 0
  local time = sys.clock()

  model:evaluate()

  -- if iter == 81 then
  --   pred_list = torch.zeros(nTesting, 1)
  --   target_list = torch.zeros(nTesting, 1)
  -- end

  print ('\n==> testing on test set:')
  for t = 1, nTesting do
    local input = testset_input[{{t, {}}}]
    input = torch.reshape(input, input:size(1), input:size(2), 1)
    if option.cuda then
      input = input:cuda()
    end
    local target = testset_target[t][1]+1
    local pred = model:forward(input)
    -- conv vs linear, I don't know why yet
    if option.feature_ex_type == 'conv' or option.feature_ex_type == 'mmconv'  then

    else
      pred = pred[1]
    end
    local err = criterion:forward(pred, target)
    f = f + err
    confusion:add(pred, target)
  end

  time = sys.clock() - time
  time = time / nTesting
  print("==> time to test 1 sample = " .. (time*1000) .. 'ms')

  print(confusion)

  f = f/nTesting

  test_accu[iter] = confusion.totalValid
  test_err[iter] = f
  print("==> current testing error = " .. f)

  cm = confusion.mat
  test_conf[iter][1] = cm[1][1]
  test_conf[iter][2] = cm[1][2]
  test_conf[iter][3] = cm[2][1]
  test_conf[iter][4] = cm[2][2]

  fscore = (cm[1][1]+cm[2][2])/(cm[1][1]+cm[1][2]+5*cm[2][1]+cm[2][2])

  -- Weight update
  -- local x = cm[1][1] + cm[2][2]
  -- local y = cm[1][2]
  -- local z = cm[2][1]
  --
  -- -- Reset criterion weight: version 1
  -- -- new_weight = torch.Tensor(4)
  -- -- new_weight[1] = 0.5*(y+5*z)/(x+y+5*z)^2
  -- -- new_weight[2] = -1*x/(x+y+5*z)^2
  -- -- new_weight[3] = -5*x/(x+y+5*z)^2
  -- -- new_weight[4] = 0.5*(y+5*z)/(x+y+5*z)^2
  --
  -- -- Reset criterion weight: version 2
  -- local A = (y+5*z)/(x+y+5*z)^2
  -- local B = -1*x/(x+y+5*z)^2
  -- local C = -5*x/(x+y+5*z)^2
  -- local alpha = (A-B)/(-1*C)
  --
  -- new_weight = torch.Tensor(4)
  -- new_weight[1] = 1
  -- new_weight[2] = 1
  -- new_weight[3] = alpha
  -- new_weight[4] = 1
  --
  -- require 'ClassNLLCriterion_HH'
  -- criterion = nn.ClassNLLCriterion_HH(new_weight)
  -- if option.cuda then
  --   criterion:cuda()
  -- end

  confusion:zero()
end
