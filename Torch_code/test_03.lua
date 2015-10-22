-- type_list = {'chal600', 'mimic600', 'chal300+mimic300', 'chal600+mimic600', 'chal600+mimic1200', 'chal600+mimic2400', 'chal600+mimicAll'}

require 'experiment_03'
experiment_03({500}, 'chal600+mimicAll', 'max')
experiment_03({500}, 'chal600+mimicAll', 'min')
experiment_03({500}, 'chal600+mimicAll', 'normal')
