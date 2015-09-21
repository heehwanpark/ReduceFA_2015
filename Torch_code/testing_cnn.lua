function test()
  local f = 0
  local time = sys.clock()

  model:evaluate()

  print ('\n==> testing on test set:')
  for t = 1, nTesting do
    local input = testset_input[{{t, {}}}]
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
