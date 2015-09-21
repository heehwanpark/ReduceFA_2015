require 'torch'
require 'cutorch'
require 'cunn'

function buildCNNModel(option)
  print '==> Construct CNN model'

  torch.manualSeed(option.weightseed)

  model = nn.Sequential()

  -- CNN
  for i = 1, option.convlayer_num do
    if i == 1 then
      -- 1st convolution layer
      model:add(nn.SpatialConvolutionMM(option.nInputFeature, option.nFeatures_c, 1, option.kernel))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(1, option.pool))
      -- Calculate # of outputs
      nConvOut = math.floor((option.inputSize - option.kernel + 1)/option.pool)
    else
      -- 2nd+ convolution layer
      model:add(nn.SpatialConvolutionMM(option.nFeatures_c, option.nFeatures_c, 1, option.kernel))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(1, option.pool))
      -- Calculate # of outputs
      nConvOut = math.floor((nConvOut - option.kernel + 1)/option.pool)
    end
  end

  -- Standard MLP
  model:add(nn.Reshape(option.nFeatures_c*nConvOut*1))

  for i = 1, option.mlplayer_num do
    if i == 1 then
      model:add(nn.Linear(option.nFeatures_c*nConvOut*1, option.nUnit_mlp))
      model:add(nn.ReLU())
      model:add(nn.Dropout(option.dropout_rate))
    else
      model:add(nn.Linear(option.nUnit_mlp, option.nUnit_mlp))
      model:add(nn.ReLU())
      model:add(nn.Dropout(option.dropout_rate))
    end
  end

  model:add(nn.Linear(option.nUnit_mlp, option.nTarget))
  model:add(nn.LogSoftMax())
  model:cuda()
  
  print(model)
  return model
end
