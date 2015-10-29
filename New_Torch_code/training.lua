function training()
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
                      output = output[1]
                      local err = criterion:forward(output, targets[i])
                      f = f + err
                      local df_do = criterion:backward(output, targets[i])
                      model:backward(inputs[i], df_do)

                      -- confusion:add(output, targets[i])
                    end

                    gradParameters:div(#inputs)
                    f = f/#inputs
                    return f, gradParameters
                  end

    _,fs = optimMethod(feval, parameters, optimState)
    cur_err = cur_err + fs[1]

    require 'test'
    test()
    upiter = upiter + 1
    model:training()
  end

  time = sys.clock() - time
  print("==> time to learn 1 epoch = " .. (time*1000) .. 'ms')

  -- print(confusion)
  cur_err = cur_err/math.floor(nTraining/batchsize)
  print("==> current training error = " .. cur_err)

  -- confusion:zero()
end
