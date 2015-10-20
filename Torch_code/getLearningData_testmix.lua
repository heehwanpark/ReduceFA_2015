require 'hdf5'

function getLearningData_testmix(option)
  local inputSize = option.inputSize
  local datatype = option.datatype
  local seed = option.dbseed

  -- datatype: chal, mimic+chal_all, mimic+chal_small
  print '==> Load datasets'

  local chal_file = hdf5.open('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5', 'r')
  local chal_input = chal_file:read('/inputs'):all()
  local chal_target = chal_file:read('/targets'):all()
  chal_file:close()

  chal_input = chal_input:transpose(1,2)
  chal_target = chal_target:transpose(1,2)

  local mimic2_file = hdf5.open('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_v3.h5', 'r')
  local mimic2_input = mimic2_file:read('/inputs'):all()
  local mimic2_target = mimic2_file:read('/targets'):all()
  mimic2_file:close()

  mimic2_input = mimic2_input:transpose(1,2)
  mimic2_target = mimic2_target:transpose(1,2)

  local chal_testsize = 128;
  local mimic_testsize = 128;
  local nEle_chal = chal_target:size(1)
  local nEle_mimic2 = mimic2_target:size(1)

  -- Fix random seed
  torch.manualSeed(seed)
  local shuffle_chal = torch.randperm(nEle_chal)
  local testindex_chal = shuffle_chal[{{1,chal_testsize}}]
  local trainindex_chal = shuffle_chal[{{chal_testsize+1,-1}}]

  torch.manualSeed(seed)
  local shuffle_mimic = torch.randperm(nEle_mimic2)
  local testindex_mimic = shuffle_mimic[{{1,mimic_testsize}}]
  local trainindex_mimic = shuffle_mimic[{{mimic_testsize+1,-1}}]
  -------------------------------------------------------------
  nTesting = chal_testsize + mimic_testsize
  testset_input = torch.zeros(nTesting, inputSize)
  testset_target = torch.zeros(nTesting, 1)
  for i = 1, nTesting do
    if i <= chal_testsize then
      testset_input[{i, {}}] = chal_input[{testindex_chal[i], {}}]
      testset_target[i] = chal_target[testindex_chal[i]]
    else
      testset_input[{i, {}}] = mimic2_input[{testindex_mimic[i-chal_testsize], {}}]
      testset_target[i] = mimic2_target[testindex_mimic[i-chal_testsize]]
    end
  end
  -------------------------------------------------------------
  -- nTesting = chal_testsize
  -- testset_input = torch.zeros(nTesting, inputSize)
  -- testset_target = torch.zeros(nTesting, 1)
  -- for i = 1, nTesting do
  --   testset_input[{i, {}}] = chal_input[{testindex_chal[i], {}}]
  --   testset_target[i] = chal_target[testindex_chal[i]]
  -- end
  -------------------------------------------------------------
  -- Data: 'chal600', 'mimic600', 'chal300+mimic300', 'chal600+mimic600', 'chal600+mimic1200', 'chal600+mimic2400', 'chal600+mimicAll'
  if datatype == 'chal600' then
    nTraining = 600
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      trainset_input[{i, {}}] = chal_input[{trainindex_chal[i], {}}]
      trainset_target[i] = chal_target[trainindex_chal[i]]
    end
  elseif datatype == 'mimic600' then
    nTraining = 600
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      trainset_input[{i, {}}] = mimic2_input[{trainindex_mimic[i], {}}]
      trainset_target[i] = mimic2_target[trainindex_mimic[i]]
    end
  elseif datatype == 'mimicAll' then
    nTraining = nEle_mimic2 - 128
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      trainset_input[{i, {}}] = mimic2_input[{trainindex_mimic[i], {}}]
      trainset_target[i] = mimic2_target[trainindex_mimic[i]]
    end
  elseif datatype == 'chal300+mimic300' then
    nTraining = 300 + 300
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      if i <= 300 then
        trainset_input[{i, {}}] = chal_input[{trainindex_chal[i], {}}]
        trainset_target[i] = chal_target[trainindex_chal[i]]
      else
        trainset_input[{i, {}}] = mimic2_input[{trainindex_mimic[i-300], {}}]
        trainset_target[i] = mimic2_target[trainindex_mimic[i-300]]
      end
    end
  elseif datatype == 'chal600+mimic600' then
    nTraining = 600 + 600
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      if i <= 600 then
        trainset_input[{i, {}}] = chal_input[{trainindex_chal[i], {}}]
        trainset_target[i] = chal_target[trainindex_chal[i]]
      else
        trainset_input[{i, {}}] = mimic2_input[{trainindex_mimic[i-600], {}}]
        trainset_target[i] = mimic2_target[trainindex_mimic[i-600]]
      end
    end
  elseif datatype == 'chal600+mimic1200' then
    nTraining = 600 + 1200
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      if i <= 600 then
        trainset_input[{i, {}}] = chal_input[{trainindex_chal[i], {}}]
        trainset_target[i] = chal_target[trainindex_chal[i]]
      else
        trainset_input[{i, {}}] = mimic2_input[{trainindex_mimic[i-600], {}}]
        trainset_target[i] = mimic2_target[trainindex_mimic[i-600]]
      end
    end
  elseif datatype == 'chal600+mimic2400' then
    nTraining = 600 + 2400
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      if i <= 600 then
        trainset_input[{i, {}}] = chal_input[{trainindex_chal[i], {}}]
        trainset_target[i] = chal_target[trainindex_chal[i]]
      else
        trainset_input[{i, {}}] = mimic2_input[{trainindex_mimic[i-600], {}}]
        trainset_target[i] = mimic2_target[trainindex_mimic[i-600]]
      end
    end
  elseif datatype == 'chal600+mimicAll' then
    nTraining = 600 + (nEle_mimic2 - 128)
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      if i <= 600 then
        trainset_input[{i, {}}] = chal_input[{trainindex_chal[i], {}}]
        trainset_target[i] = chal_target[trainindex_chal[i]]
      else
        trainset_input[{i, {}}] = mimic2_input[{trainindex_mimic[i-600], {}}]
        trainset_target[i] = mimic2_target[trainindex_mimic[i-600]]
      end
    end
  else
    print("Wrong dataset type!")
    do return end
  end
  return nTraining, trainset_input, trainset_target, nTesting, testset_input, testset_target
end
