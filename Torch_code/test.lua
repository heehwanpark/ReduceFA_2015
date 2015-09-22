require 'experiment_01_diffweight'

seed_list = {1, 2, 5, 10, 25}
type_list = {'chal', 'mimic+chal_all', 'mimic+chal_small', 'mimic2_all', 'mimic2_small'}

experiment_01('mimic+chal_all', 1, 1)

-- for i = 1, 1 do
--   print(type_list[1] .. '-' .. seed_list[i])
--   experiment_01(type_list[4], seed_list[1], seed_list[1])
--   experiment_01(type_list[5], seed_list[1], seed_list[1])
-- end
