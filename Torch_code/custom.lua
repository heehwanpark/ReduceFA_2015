function arch2string(architecture)
  arch_str = '['
  for i = 1, table.getn(architecture) do
    if i == table.getn(architecture) then
      arch_str = arch_str .. architecture[i] .. ']'
    else
      arch_str = arch_str .. architecture[i] .. '-'
    end
  end
  return arch_str
end

function maxFilter(dataset)
  nEle = dataset:size(1)
  old_inputSize = dataset:size(2)
  window = option.mwindow
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

function minFilter(dataset)
  nEle = dataset:size(1)
  old_inputSize = dataset:size(2)
  window = option.mwindow
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
