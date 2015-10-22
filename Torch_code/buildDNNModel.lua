require 'torch'
require 'cutorch'
require 'cunn'

function buildDNNModel(option)
  print '==> Construct DNN model'

  mlp_architecture = option.mlp_architecture
  nHiddenlayers = table.getn(mlp_architecture)

  torch.manualSeed(option.weightseed)

  model = nn.Sequential()
  model:add(nn.Reshape(option.inputSize))

  for i = 1, nHiddenlayers do
    if i == 1 then
      model:add(nn.Linear(option.inputSize, mlp_architecture[i]))
      model:add(nn.ReLU())
    else
      model:add(nn.Linear(mlp_architecture[i-1], mlp_architecture[i]))
      model:add(nn.ReLU())
    end
    if option.dropout_rate > 0 and option.dropout_rate < 1 then
      model:add(nn.Dropout(option.dropout_rate))
    end
  end

  model:add(nn.Linear(mlp_architecture[nHiddenlayers], option.nTarget))
  model:add(nn.LogSoftMax())
  model:cuda()

  print(model)
  return model
end
