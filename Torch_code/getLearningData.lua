require 'hdf5'

function getLearningData(option)
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

  -- Fix random seed
  torch.manualSeed(seed)

  -- # of training elements in challenge 2015 set
  local chal_trainsize = 600
  local nEle_chal = chal_target:size(1)
  local nEle_mimic2 = mimic2_target:size(1)
  local shuffle = torch.randperm(nEle_chal)

  nTesting = nEle_chal - chal_trainsize
  testset_idx = shuffle[{{1,nTesting}}]
  testset_input = torch.zeros(nTesting, inputSize)
  testset_target = torch.zeros(nTesting, 1)
  for i = 1, nTesting do
    testset_input[{i, {}}] = chal_input[{shuffle[i], {}}]
    testset_target[i] = chal_target[shuffle[i]]
  end

  if datatype == 'chal' then
    -- Make training set: challenge2015
    nTraining = chal_trainsize
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      trainset_input[{i, {}}] = chal_input[{shuffle[nTesting+i], {}}]
      trainset_target[i] = chal_target[shuffle[nTesting+i]]
    end
  elseif datatype == 'mimic+chal_all' then
    -- Make training set: mimic2 + challenge2015
    nTraining = nEle_mimic2 + chal_trainsize
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      if i <= nEle_mimic2 then
        trainset_input[{i, {}}] = mimic2_input[{i, {}}]
        trainset_target[i] = mimic2_target[i]
      else
        trainset_input[{i, {}}] = chal_input[{shuffle[nTesting+i-(nEle_mimic2)], {}}]
        trainset_target[i] = chal_target[shuffle[nTesting+i-(nEle_mimic2)]]
      end
    end
  elseif datatype == 'mimic+chal_small' then
    shuffle_mimic = torch.randperm(nEle_mimic2)
    shuffle_chal = torch.randperm(nEle_chal)

    nTraining = chal_trainsize
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      if i <= nTraining/2 then
        trainset_input[{i, {}}] = mimic2_input[{shuffle_mimic[i], {}}]
        trainset_target[i] = mimic2_target[shuffle_mimic[i]]
      else
        trainset_input[{i, {}}] = chal_input[{shuffle_chal[i], {}}]
        trainset_target[i] = chal_target[shuffle_chal[i]]
      end
    end
  elseif datatype == 'mimic2_all' then
    nTraining = nEle_mimic2
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      trainset_input[{i, {}}] = mimic2_input[{i, {}}]
      trainset_target[i] = mimic2_target[i]
    end
  elseif datatype == 'mimic2_small' then
    local shuffle_all = torch.randperm(nEle_mimic2)
    nTraining = chal_trainsize
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      trainset_input[{i, {}}] = mimic2_input[{shuffle_all[i], {}}]
      trainset_target[i] = mimic2_target[shuffle_all[i]]
    end
  else
    print("Wrong dataset type!")
    do return end
  end
  return nTraining, trainset_input, trainset_target, nTesting, testset_input, testset_target
end
