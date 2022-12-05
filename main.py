import powerlaw as pl
import os

dir_parameters = 'parameter_settings'
files = os.listdir(dir_parameters)
n = 1
dir_output = '.'
pl.run_iterations_withParallel(files,n,dir_parameters,100,1,dir_output,'observed_network',1)
