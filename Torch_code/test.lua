-- require 'experiment_01_diffweight'

seed_list = {1, 2, 5, 10, 25}
type_list = {'chal', 'mimic+chal_all', 'mimic+chal_small', 'mimic2_all', 'mimic2_small'}

require 'experiment_01_max'
experiment_01('05_max', type_list[2], seed_list[1], seed_list[1])

-- for i = 1, 5 do
--   experiment_01('01_data', type_list[i], seed_list[1], seed_list[1])
--   experiment_01('02_initweight', type_list[2], seed_list[1], seed_list[i])
--   experiment_01('03_difftest', type_list[2], seed_list[i], seed_list[1])
-- end
