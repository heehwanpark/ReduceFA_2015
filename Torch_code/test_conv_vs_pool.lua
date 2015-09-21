require 'torch'
require 'detectFAinICU_v1'

torch.manualSeed(1) -- 1, 100, 200
datatype = 'chal2015'
inputSize = 2500
x = torch.Tensor(10,10)
s = x:storage()

condition = {1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,0,0,0,0,
            1,1,1,1,0,0,0,0,0,0,
            1,1,1,1,0,0,0,0,0,0,
            1,1,1,0,0,0,0,0,0,0,
            1,1,1,0,0,0,0,0,0,0,
            1,1,1,0,0,0,0,0,0,0,
            1,1,1,0,0,0,0,0,0,0,
            1,1,0,0,0,0,0,0,0,0,
            1,1,0,0,0,0,0,0,0,0}

for i=1, s:size() do
  s[i] = condition[i]
end
x = x:transpose(1,2)
print(x)

-----------------------------------------------------
print '==> Load datasets'
require 'hdf5'

if datatype == 'both' then
  chal_file = hdf5.open('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5', 'r')
  chal_input = chal_file:read('/inputs'):all()
  chal_target = chal_file:read('/targets'):all()
  chal_file:close()

  chal_input = chal_input:transpose(1,2)
  chal_target = chal_target:transpose(1,2)

  mimic2_file = hdf5.open('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_mimic2_v3.h5', 'r')
  mimic2_input = mimic2_file:read('/inputs'):all()
  mimic2_target = mimic2_file:read('/targets'):all()
  mimic2_file:close()

  mimic2_input = mimic2_input:transpose(1,2)
  mimic2_target = mimic2_target:transpose(1,2)

  nSample_chal = chal_target:size(1)
  nSample_mimic2 = mimic2_target:size(1)
  -- nSample_mitdb = mitdb_target:size(1)

  nElement = nSample_chal + nSample_mimic2
  -- nElement = nSample_chal + nSample_mimic2
  chal_nTraining = 600
  nTesting = nSample_chal - chal_nTraining
  nTraining = nElement - nTesting

  shuffle = torch.randperm(nSample_chal)

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
elseif datatype == 'chal2015' then
  chal_file = hdf5.open('/home/heehwan/Workspace/Data/ReduceFA_2015/cnndb_chal2015.h5', 'r')
  chal_input = chal_file:read('/inputs'):all()
  chal_target = chal_file:read('/targets'):all()
  chal_file:close()

  chal_input = chal_input:transpose(1,2)
  chal_target = chal_target:transpose(1,2)

  nElement = chal_target:size(1)
  nTraining = 600
  nTesting = nElement - nTraining

  shuffle = torch.randperm(nElement)

  -- Make training set: mimic2 + challenge2015
  trainset_input = torch.zeros(nTraining, inputSize)
  trainset_target = torch.zeros(nTraining, 1)
  for i = 1, nTraining do
    trainset_input[{i, {}}] = chal_input[{shuffle[i], {}}]
    trainset_target[i] = chal_target[shuffle[i]]
  end
  -- Make test set: mimic2 + challenge2015
  testset_input = torch.zeros(nTesting, inputSize)
  testset_target = torch.zeros(nTesting, 1)
  for i = 1, nTesting do
    testset_input[{i, {}}] = chal_input[{shuffle[i+nTraining], {}}]
    testset_target[i] = chal_target[shuffle[i+nTraining]]
  end
end

print(trainset_input:size())
print(testset_input:size())

detectFAinICU_v1(3, 5, datatype .. '0918_01')

-- for i=4,10 do
--   for j=1,10 do
--     if x[i][j] == 1 then
--       print('###############################')
--       print('## # of Conv ' .. i .. ' && pool size ' .. j .. '##')
--       print('###############################')
--
--       index = i .. '-' .. j
--       detectFAinICU_v1(i, j, index)
--     end
--   end
-- end
