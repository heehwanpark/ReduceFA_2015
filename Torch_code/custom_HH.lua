
function normalize_HH(data)
  N = data:size(1)
  for i = 1,N do
    mean = torch.mean(data[{i,{}}])
    std = torch.std(data[{i,{}}])
    data[{i,{}}] = data[{i,{}}] - mean
  end
end

function convertForPretrain(dataset, inputSize)
  N = dataset:size(1)
  converted = {}
  for i = 1, N do
    data = dataset[{i, {}}]
    -- print(data[1]:size())
    data = torch.reshape(data, 1, inputSize, 1)
    table.insert(converted, data)
  end
  return converted
end

function netsThrough(model, dataset)
  print(dataset:size())
  N = table.getn(dataset)
  outputs = {}
  for i = 1, N do
    input = dataset[i]
    print(input:size())
    -- input = torch.reshape(input, 1, input:size(), 1)
    output = model:forward(input)
    table.insert(outputs, output)
  end
  return outputs
end

function makeMax(dataset)
  nEle = dataset:size(1)
  old_inputSize = dataset:size(2)
  window = option.maxmin_window
  new_inputSize = old_inputSize - window + 1

  new_dataset = torch.zeros(nEle, new_inputSize)
  for i = 1, nEle do
    old_input = dataset[{i, {}}]
    for j = 1, new_inputSize do
      local val = torch.max(old_input[{{j,j+window-1}}])
      new_dataset[i][j] = val
    end
  end

  return new_dataset
end

function makeMin(dataset)
  nEle = dataset:size(1)
  old_inputSize = dataset:size(2)
  window = option.maxmin_window
  new_inputSize = old_inputSize - window + 1

  new_dataset = torch.zeros(nEle, new_inputSize)
  for i = 1, nEle do
    old_input = dataset[{i, {}}]
    for j = 1, new_inputSize do
      local val = torch.min(old_input[{{j,j+window-1}}])
      new_dataset[i][j] = val
    end
  end

  return new_dataset
end
-- function netsThrough(model, dataset)
--   N = dataset:size(1)
--   outputs = {}
--   for i = 1, N do
--     input = dataset[{i, {}}]
--     input = torch.reshape(input, 1, input:size(), 1)
--     output = model:forward(input)
--     table.insert(outputs, output)
--   end
--   return outputs
-- end
