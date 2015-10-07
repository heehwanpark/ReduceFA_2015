type_list = {'chal600', 'mimic600', 'chal300+mimic300', 'chal600+mimic600', 'chal600+mimic1200', 'chal600+mimic2400', 'chal600+mimicAll'}

require 'experiment_02'
experiment_02('01_data', type_list[7])
-- for i = 1, 7 do
--   experiment_02('01_data', type_list[i])
-- end
