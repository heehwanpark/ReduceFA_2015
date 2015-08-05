
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
  N = table.getn(dataset)
  outputs = {}
  for i = 1, N do
    input = dataset[i]
    -- input = torch.reshape(input, 1, input:size(), 1)
    output = model:forward(input)
    table.insert(outputs, output)
  end
  return outputs
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
