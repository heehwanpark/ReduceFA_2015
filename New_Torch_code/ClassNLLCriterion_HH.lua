local ClassNLLCriterion_HH, parent = torch.class(
    'nn.ClassNLLCriterion_HH',
    'nn.Criterion'
)

function ClassNLLCriterion_HH:__init(weights, sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
    if weights then
        assert(weights:dim() == 1, "weights input should be 1-D Tensor")
        self.weights = weights
    end

    self.output_tensor = torch.zeros(1)
    self.total_weight_tensor = torch.zeros(1)
    self.target = torch.zeros(1):long()
end

function ClassNLLCriterion_HH:__len()
   if (self.weights) then
      return #self.weights
   else
      return 0
   end
end

function ClassNLLCriterion_HH:updateOutput(input, target)
    if type(target) == 'number' then
        self.target[1] = target
    elseif target:type() == 'torch.CudaTensor' then
        self.target = target
    else
        self.target = target:long()
    end

    if input:size(1) == 2 and input:dim() == 1 then
      if target == 1 then
        if input[1] > input[2] then
          self.output = -input[target] * self.weights[4]
        else
          self.output = -input[target] * self.weights[2]
        end
      else
        if input[1] > input[2] then
          self.output = -input[target] * self.weights[3]
        else
          self.output = -input[target] * self.weights[1]
        end
      end
    else
      error('vector of 2 elements expected')
    end

    return self.output
end

function ClassNLLCriterion_HH:updateGradInput(input, target)
    if type(target) == 'number' then
        self.target[1] = target
    elseif target:type() == 'torch.CudaTensor' then
        self.target = target
    else
        self.target = target:long()
    end

    self.gradInput:resizeAs(input):zero()

    self.gradInput[target] = -1
    if target == 1 then
      if input[1] > input[2] then
        self.gradInput[target] = self.gradInput[target] * self.weights[4]
      else
        self.gradInput[target] = self.gradInput[target] * self.weights[2]
      end
    else
      if input[1] > input[2] then
        self.gradInput[target] = self.gradInput[target] * self.weights[3]
      else
        self.gradInput[target] = self.gradInput[target] * self.weights[1]
      end
    end

    return self.gradInput
end
