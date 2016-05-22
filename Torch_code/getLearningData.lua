require 'hdf5'

function getLearningData(option)
  local trdata_type = option.trdata_type
  local testdata_type = option.testdata_type

  print '==> Load datasets'

  if option.feature_ex_type == 'gauss' then
    chal_file = hdf5.open('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015_gauss.h5', 'r')
  else
    chal_file = hdf5.open('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5', 'r')
  end
  local chal_input = chal_file:read('/inputs'):all()
  local chal_target = chal_file:read('/targets'):all()
  chal_file:close()

  chal_input = chal_input:transpose(1,2)
  chal_target = chal_target:transpose(1,2)

  if option.feature_ex_type == 'gauss' then
    mimic2_file = hdf5.open('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_gauss.h5', 'r')
  else
    mimic2_file = hdf5.open('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_v3.h5', 'r')
  end
  local mimic2_input = mimic2_file:read('/inputs'):all()
  local mimic2_target = mimic2_file:read('/targets'):all()
  mimic2_file:close()

  mimic2_input = mimic2_input:transpose(1,2)
  mimic2_target = mimic2_target:transpose(1,2)

  local nEle_chal = chal_target:size(1)
  local nEle_mimic2 = mimic2_target:size(1)

  torch.manualSeed(option.db_seed)
  local shuffle_chal = torch.randperm(nEle_chal)
  local chal_testsize = 128
  local testindex_chal = shuffle_chal[{{1,chal_testsize}}]
  local trainindex_chal = shuffle_chal[{{chal_testsize+1,-1}}]

  torch.manualSeed(option.db_seed)
  local shuffle_mimic = torch.randperm(nEle_mimic2)
  if testdata_type == 'chal+mimic' then
    mimic_testsize = 128
    testindex_mimic = shuffle_mimic[{{1,mimic_testsize}}]
    trainindex_mimic = shuffle_mimic[{{mimic_testsize+1,-1}}]
  elseif testdata_type == 'chal' then
    mimic_testsize = 0
    trainindex_mimic = shuffle_mimic
  end

  -- test_result = hdf5.open(foldername .. filename .. '_test_result.h5', 'w')
  -- test_result:write('/testindex_chal', testindex_chal)
  -- if testdata_type == 'chal+mimic' then
  --   test_result:write('/testindex_mimic', testindex_mimic)
  -- end
  -------------------------------------------------------------
  if option.feature_ex_type == 'gauss' then
    option.inputSize = chal_input:size(2)
  end

  nTesting = chal_testsize + mimic_testsize
  testset_input = torch.zeros(nTesting, option.inputSize)
  testset_target = torch.zeros(nTesting, 1)

  testnum_chal = 0
  testnum_mimic = 0
  for i = 1, nTesting do
    if i <= chal_testsize then
      testset_input[{i, {}}] = chal_input[{testindex_chal[i], {}}]
      testset_target[i] = chal_target[testindex_chal[i]]
      testnum_chal = testnum_chal + chal_target[testindex_chal[i]][1]
    else
      testset_input[{i, {}}] = mimic2_input[{testindex_mimic[i-chal_testsize], {}}]
      testset_target[i] = mimic2_target[testindex_mimic[i-chal_testsize]]
      testnum_mimic = testnum_mimic + mimic2_target[testindex_mimic[i-chal_testsize]][1]
    end
  end

  print('==> test num')
  print(testnum_chal)
  print(testnum_mimic)
  -------------------------------------------------------------

  trnum_chal = 0
  trnum_mimic = 0

  if trdata_type == 'chal600+mimicAll' then
    nTraining = (nEle_chal-chal_testsize) + (nEle_mimic2-mimic_testsize)
    trainset_input = torch.zeros(nTraining, option.inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      if i <= (nEle_chal-chal_testsize) then
        trainset_input[{i, {}}] = chal_input[{trainindex_chal[i], {}}]
        trainset_target[i] = chal_target[trainindex_chal[i]]
        trnum_chal = trnum_chal + chal_target[trainindex_chal[i]][1]
      else
        local idx = i-(nEle_chal-chal_testsize)
        trainset_input[{i, {}}] = mimic2_input[{trainindex_mimic[idx], {}}]
        trainset_target[i] = mimic2_target[trainindex_mimic[idx]]
        trnum_mimic = trnum_mimic + mimic2_target[trainindex_mimic[idx]][1]
      end
    end
  elseif trdata_type == 'chal600' then
    nTraining = nEle_chal-chal_testsize;
    trainset_input = torch.zeros(nTraining, option.inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      trainset_input[{i, {}}] = chal_input[{trainindex_chal[i], {}}]
      trainset_target[i] = chal_target[trainindex_chal[i]]
    end
  else
    print("Wrong dataset type!")
    do return end
  end
  print('==> train num')
  print(trnum_chal)
  print(trnum_mimic)
  return nTraining, trainset_input, trainset_target, nTesting, testset_input, testset_target
end
