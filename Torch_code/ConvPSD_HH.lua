function trainConvPSD(pretrainset, nInputFeature, nOutputFeature, option, filename)
  ----------------------------------------------------------------------
  print '==> construct Pre-training model'

  -- connection table:
  conntable = nn.tables.full(nInputFeature, nOutputFeature)
  -- decoder's table:
  local decodertable = conntable:clone()
  decodertable[{ {},1 }] = conntable[{ {},2 }]
  decodertable[{ {},2 }] = conntable[{ {},1 }]
  local outputSize = conntable[{ {},2 }]:max()

  encoder = nn.Sequential()
  encoder:add(nn.SpatialConvolutionMap(conntable, 1, option.kernel))
  encoder:add(nn.ReLU())
  encoder:add(nn.Diag(outputSize))

  decoder = unsup.SpatialConvFistaL1(decodertable, 1, option.kernel, 1, option.inputSize, option.lambda)

  -- module
  module = unsup.PSD(encoder, decoder, option.beta)
  ----------------------------------------------------------------------
  print '==> defining Pre-training procedure'

  -- nPretrain = pretrainset:size(1)
  nPretrain = table.getn(pretrainset)
  batchsize = option.batchsize
  x, dl_dx = module:getParameters()

  sgdconf = {learningRate = option.lr_unsup}

  Maxiter = 1
  local count = 1

  while count <= Maxiter do

    local time = sys.clock()

    print('==> doing epoch on pre-training data:')
    print("==> online epoch # " .. count .. ' [batchSize = ' .. batchsize .. ']')

    local cur_err = 0;

    for t = 1, math.floor(nPretrain/batchsize) do
      samples = {}
      batchstart = (t-1)*batchsize+1
      for i = batchstart, batchstart+batchsize-1 do
        -- local input = pretrainset[{i, {}}]
        -- input = torch.reshape(input, 1, input:size(1), 1)
        local input = pretrainset[i]
        table.insert(samples, input)
      end

      local feval = function()

        local f = 0
        dl_dx:zero()

        for i = 1, batchsize do
          f = f + module:updateOutput(samples[i], samples[i])
          module:updateGradInput(samples[i], samples[i])
          module:accGradParameters(samples[i], samples[i])
        end

        dl_dx:div(batchsize)
        f = f/batchsize

        return f, dl_dx
      end

      x,fx = optim.sgd(feval, x, sgdconf)
      print(fx[1])
      cur_err = cur_err + fx[1]

      module:normalize()
    end

    print("\n==> current error = " .. (cur_err/math.floor(nPretrain/batchsize)))

    time = sys.clock() - time
    time = time / nPretrain

    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- module:normalize()

    count = count + 1
  end

  encoder_filename = filename .. "_encoder.net"
  decoder_filename = filename .. "_decoder.net"

  torch.save(encoder_filename, module.encoder)
  torch.save(decoder_filename, module.decoder)

  return module.encoder, module.decoder
end
