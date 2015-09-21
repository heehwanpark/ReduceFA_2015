require 'experiment_01_diffweight'

seed_list = {1, 2, 5, 10, 25}
type_list = {'chal', 'mimic+chal_all', 'mimic+chal_small'}

for i = 1, 5 do
  experiment_01(type_list[1], seed_list[i])
  experiment_01(type_list[3], seed_list[i])
end
