require 'hdf5'

function getLearningData(option)
  local trdata_type = option.trdata_type
  local testdata_type = option.testdata_type

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

  test_result = hdf5.open(foldername .. filename .. '_test_result.h5', 'w')
  test_result:write('/testindex_chal', testindex_chal)
  test_result:write('/testindex_mimic', testindex_mimic)
  -------------------------------------------------------------
  nTesting = chal_testsize + mimic_testsize
  testset_input = torch.zeros(nTesting, option.inputSize)
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
  if trdata_type == 'chal600+mimicAll' then
    nTraining = (nEle_chal-chal_testsize) + (nEle_mimic2-mimic_testsize)
    trainset_input = torch.zeros(nTraining, option.inputSize)
    trainset_target = torch.zeros(nTraining, 1)
    for i = 1, nTraining do
      if i <= (nEle_chal-chal_testsize) then
        trainset_input[{i, {}}] = chal_input[{trainindex_chal[i], {}}]
        trainset_target[i] = chal_target[trainindex_chal[i]]
      else
        local idx = i-(nEle_chal-chal_testsize)
        trainset_input[{i, {}}] = mimic2_input[{trainindex_mimic[idx], {}}]
        trainset_target[i] = mimic2_target[trainindex_mimic[idx]]
      end
    end
  else
    print("Wrong dataset type!")
    do return end
  end
  return nTraining, trainset_input, trainset_target, nTesting, testset_input, testset_target
end
