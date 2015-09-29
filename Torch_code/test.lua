-- require 'experiment_01_diffweight'

seed_list = {1, 2, 5, 10, 25}
type_list = {'chal', 'mimic+chal_all', 'mimic+chal_small', 'mimic2_all', 'mimic2_small'}
max_list = {50, 100, 150, 200, 250, 300}

require 'experiment_01_max'
for i = 1, 6 do
  experiment_01('05_max', type_list[2], seed_list[1], seed_list[1], max_list[i])
end

require 'experiment_01_wavelet'
for i = 1, 5 do
  experiment_01_wavelet(seed_list[i])
end
-- for i = 1, 5 do
--   experiment_01('01_data', type_list[i], seed_list[1], seed_list[1])
--   experiment_01('02_initweight', type_list[2], seed_list[1], seed_list[i])
--   experiment_01('03_difftest', type_list[2], seed_list[i], seed_list[1])
-- end
