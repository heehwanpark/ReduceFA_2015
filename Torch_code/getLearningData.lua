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
  chal_trainsize = 600

  if datatype == 'chal' then
    nElement = chal_target:size(1)
    nTraining = chal_trainsize
    nTesting = nElement - nTraining

    local shuffle = torch.randperm(nElement)

    -- Make training set: challenge2015
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      trainset_input[{i, {}}] = chal_input[{shuffle[i], {}}]
      trainset_target[i] = chal_target[shuffle[i]]
    end
    -- Make test set: challenge2015
    testset_input = torch.zeros(nTesting, inputSize)
    testset_target = torch.zeros(nTesting, 1)
    for i = 1, nTesting do
      testset_input[{i, {}}] = chal_input[{shuffle[i+nTraining], {}}]
      testset_target[i] = chal_target[shuffle[i+nTraining]]
    end
  elseif datatype == 'mimic+chal_all' then
    local nSample_chal = chal_target:size(1)
    local nSample_mimic2 = mimic2_target:size(1)

    local chal_nTraining = chal_trainsize
    nElement = nSample_chal + nSample_mimic2
    nTesting = nSample_chal - chal_nTraining
    nTraining = nElement - nTesting

    local shuffle = torch.randperm(nSample_chal)

    -- Make training set: mimic2 + challenge2015
    trainset_input = torch.zeros(nTraining, inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      if i <= nSample_mimic2 then
        trainset_input[{i, {}}] = mimic2_input[{i, {}}]
        trainset_target[i] = mimic2_target[i]
      else
        trainset_input[{i, {}}] = chal_input[{shuffle[i-(nSample_mimic2)], {}}]
        trainset_target[i] = chal_target[shuffle[i-(nSample_mimic2)]]
      end
    end
    -- Make test set: mimic2 + challenge2015
    testset_input = torch.zeros(nTesting, inputSize)
    testset_target = torch.zeros(nTesting, 1)
    for i = 1, nTesting do
      testset_input[{i, {}}] = chal_input[{shuffle[i+chal_nTraining], {}}]
      testset_target[i] = chal_target[shuffle[i+chal_nTraining]]
    end
  elseif datatype == 'mimic+chal_small' then
    local nSample_chal = chal_target:size(1)
    local nSample_mimic2 = mimic2_target:size(1)

    local chal_nTraining = chal_trainsize
    nElement = nSample_chal + nSample_mimic2
    nTesting = nSample_chal - chal_nTraining
    nTraining = nElement - nTesting

    local shuffle = torch.randperm(nSample_chal)

    -- Make training set: mimic2 + challenge2015
    local all_trainset_input = torch.zeros(nTraining, inputSize)
    local all_trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      if i <= nSample_mimic2 then
        all_trainset_input[{i, {}}] = mimic2_input[{i, {}}]
        all_trainset_target[i] = mimic2_target[i]
      else
        all_trainset_input[{i, {}}] = chal_input[{shuffle[i-(nSample_mimic2)], {}}]
        all_trainset_target[i] = chal_target[shuffle[i-(nSample_mimic2)]]
      end
    end

    local shuffle_all = torch.randperm(nTraining)

    trainset_input = torch.zeros(chal_trainsize, inputSize)
    trainset_target = torch.zeros(chal_trainsize, 1)
    for i = 1, chal_trainsize do
      trainset_input[{i, {}}] = all_trainset_input[{shuffle_all[i], {}}]
      trainset_target[i] = all_trainset_target[shuffle_all[i]]
    end

    testset_input = torch.zeros(nTesting, inputSize)
    testset_target = torch.zeros(nTesting, 1)
    for i = 1, nTesting do
      testset_input[{i, {}}] = chal_input[{shuffle[i+chal_nTraining], {}}]
      testset_target[i] = chal_target[shuffle[i+chal_nTraining]]
    end
  else
    print("Wrong dataset type!")
    do return end
  end

  return nTraining, trainset_input, trainset_target, nTesting, testset_input, testset_target
end
