function train()
  local cur_err = 0
  local time = sys.clock()
  shuffle_t = torch.randperm(nTraining)

  print('\n==> doing epoch on training data:')
  print("==> online epoch # " .. iter .. ' [batchSize = ' .. batchsize .. ']')
  for t = 1, math.floor(nTraining/batchsize) do

    model:training()

    local inputs = {}
    local targets = {}
    batchstart = (t-1)*batchsize+1
    for i = batchstart, batchstart+batchsize-1 do
      local input = trainset_input[{{shuffle_t[i], {}}}]
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
      end

      gradParameters:div(#inputs)
      f = f/#inputs
      return f, gradParameters
    end

    _,fs = optimMethod(feval, parameters, optimState)
    cur_err = cur_err + fs[1]

    require 'testing_cnn_02'
    test()

    upIter = upIter + 1
  end

  model:training()

  time = sys.clock() - time
  time = time / nTraining

  -- print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

  cur_err = cur_err/math.floor(nTraining/batchsize)
  print("==> current training error = " .. cur_err)
end
