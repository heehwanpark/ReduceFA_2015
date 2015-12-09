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

    -- if iter == 81 then
    --   y,i_y = torch.max(pred,1)
    --   pred_list[t] = i_y[1]
    --   target_list[t] = target
    -- end
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

  confusion:zero()
end
