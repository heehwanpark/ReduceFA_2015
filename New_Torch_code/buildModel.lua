require 'cutorch'
require 'cunn'

function buildModel(option)
  print '==> Construct Neural Nets model'

  torch.manualSeed(option.net_init_seed)
  model = nn.Sequential()
  if option.feature_ex_type == 'conv' then -- For convolutional networks
    for i = 1, table.getn(option.conv_architecture) do
      if i == 1 then
        -- 1st convolution layer
        model:add(nn.SpatialConvolutionMM(option.nInputFeature, option.conv_architecture[i], 1, option.conv_kernel))
        model:add(nn.ReLU())
        model:add(nn.SpatialMaxPooling(1, option.conv_pool))
        -- Calculate # of outputs
        nConvOut = math.floor((option.inputSize - option.conv_kernel + 1)/option.conv_pool)
      else
        -- 2nd+ convolution layer
        model:add(nn.SpatialConvolutionMM(option.conv_architecture[i-1], option.conv_architecture[i], 1, option.conv_kernel))
        model:add(nn.ReLU())
        model:add(nn.SpatialMaxPooling(1, option.conv_pool))
        -- Calculate # of outputs
        nConvOut = math.floor((nConvOut - option.conv_kernel + 1)/option.conv_pool)
      end
    end
    n_feature_out = option.conv_architecture[table.getn(option.conv_architecture)]*nConvOut*1
  elseif option.feature_ex_type == 'only_pool' then
    local pool_size = 2
    model:add(nn.SpatialMaxPooling(1, pool_size))
    n_feature_out = math.floor(option.inputSize/pool_size)
  elseif option.feature_ex_type == 'only_pool_x2' then
    local pool_size = 2
    model:add(nn.SpatialMaxPooling(1, pool_size))
    n_feature_out = math.floor(option.inputSize/pool_size)

    model:add(nn.SpatialMaxPooling(1, pool_size))
    n_feature_out = math.floor(n_feature_out/pool_size)
  elseif option.feature_ex_type == 'max' then
    trainset_input = maxFilter(trainset_input)
    testset_input = maxFilter(testset_input)
    n_feature_out = option.inputSize - option.mwindow + 1
  elseif option.feature_ex_type == 'min' then
    trainset_input = minFilter(trainset_input)
    testset_input = minFilter(testset_input)
    n_feature_out = option.inputSize - option.mwindow + 1
  elseif option.feature_ex_type == 'max-min' then
    trainset_input = maxFilter(trainset_input)-minFilter(trainset_input)
    testset_input = maxFilter(testset_input)-minFilter(testset_input)
    n_feature_out = option.inputSize - option.mwindow + 1
  elseif option.feature_ex_type == 'max-min_x2' then
    trainset_input = maxFilter(trainset_input)-minFilter(trainset_input)
    testset_input = maxFilter(testset_input)-minFilter(testset_input)
    n_feature_out = option.inputSize - option.mwindow + 1

    trainset_input = maxFilter(trainset_input)-minFilter(trainset_input)
    testset_input = maxFilter(testset_input)-minFilter(testset_input)
    n_feature_out = n_feature_out - option.mwindow + 1
  elseif option.feature_ex_type == 'mlp' then
    n_feature_out = option.inputSize
  elseif option.feature_ex_type == 'mmpool' then
    trainset_input = maxFilter(trainset_input)-minFilter(trainset_input)
    testset_input = maxFilter(testset_input)-minFilter(testset_input)
    local pool_size = 2
    model:add(nn.SpatialMaxPooling(1, pool_size))
    n_feature_out = math.floor((option.inputSize - option.mwindow + 1)/pool_size)
  elseif option.feature_ex_type == 'mmconv' then
    trainset_input = maxFilter(trainset_input)-minFilter(trainset_input)
    testset_input = maxFilter(testset_input)-minFilter(testset_input)
    local nMMout = option.inputSize - option.mwindow + 1
    for i = 1, table.getn(option.conv_architecture) do
      if i == 1 then
        -- 1st convolution layer
        model:add(nn.SpatialConvolutionMM(option.nInputFeature, option.conv_architecture[i], 1, option.conv_kernel))
        model:add(nn.ReLU())
        model:add(nn.SpatialMaxPooling(1, option.conv_pool))
        -- Calculate # of outputs
        nConvOut = math.floor((nMMout - option.conv_kernel + 1)/option.conv_pool)
      else
        -- 2nd+ convolution layer
        model:add(nn.SpatialConvolutionMM(option.conv_architecture[i-1], option.conv_architecture[i], 1, option.conv_kernel))
        model:add(nn.ReLU())
        model:add(nn.SpatialMaxPooling(1, option.conv_pool))
        -- Calculate # of outputs
        nConvOut = math.floor((nConvOut - option.conv_kernel + 1)/option.conv_pool)
      end
    end
    n_feature_out = option.conv_architecture[table.getn(option.conv_architecture)]*nConvOut*1
  else
    print('Something wrong!')
  end

  -- Standard MLP
  model:add(nn.Reshape(n_feature_out))
  n_mlp_layer = table.getn(option.mlp_architecture)
  for i = 1, n_mlp_layer do
    if i == 1 then
      model:add(nn.Linear(n_feature_out, option.mlp_architecture[i]))
      model:add(nn.ReLU())
    else
      model:add(nn.Linear(option.mlp_architecture[i-1], option.mlp_architecture[i]))
      model:add(nn.ReLU())
    end
    if option.dropout_rate > 0 and option.dropout_rate < 1 then
      model:add(nn.Dropout(option.dropout_rate))
    end
  end

  model:add(nn.Linear(option.mlp_architecture[n_mlp_layer], option.nTarget))
  model:add(nn.LogSoftMax())
  model:cuda()

  print(model)
  return model
end
