import pandas as pd
import itertools as itt
import ppinetsim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import networkx as nx
import seaborn as sns
from os.path import join
import time
import os
from joblib import Parallel, delayed
import json
import codecs


##### create json input files #######

seq = np.arange(0,0.45,0.05)


par = ['params_AP-MS_FPR00.json','params_Y2H_FPR00.json']

for p in par:
  fp = open('parameter_settings/'+p)
  data = json.load(fp)
  fp.close()
  
  for FP in seq:
      for FN in seq:
        for a in [0.0,0.5]:
          data['false_negative_rate'] = round(FN,4)
          data['false_positive_rate'] = round(FP,4)
          data['acceptance_threshold'] = a
          # create folder
          d = 'parameter_settings/all_param_combinations/'+ data['test_method']
          if os.path.exists(d) == False:
            os.mkdir(d)
          
          with open(d+'/params_'+data['test_method'] + '_accTh'+str(data['acceptance_threshold']).replace('.','')+'_FPR' + str(data['false_positive_rate']).replace('.','')+'_FNR'+ str(data['false_negative_rate']).replace('.','')+'.json', "w") as outfile:
            json.dump(data, outfile, indent = 4)

print('parameters files done!')
#----------------------------------------------------------------------------
# if os.path.exists('output_results') == False:
#   os.mkdir('output_results')
# 
# method = ['AP-MP','Y2H']
# nsg = 50
# 
# for m in method:
#   dir_parameters = 'parameter_settings/all_param_combinations/'+ m +'/'
#   files = os.listdir(dir_parameters)
#   
#   for f in files:
#     parameters = ppinetsim.Parameters('parameter_settings/all_param_combinations/'+m+'/'+f)
#     
#     if os.path.exists('output_results/'+str(parameters.test_method)) == False:
#         os.mkdir('output_results/'+str(parameters.test_method))
#     d = 'output_results/'+str(parameters.test_method)+'/accTh' + str(parameters.acceptance_threshold).replace('.','') + '_FPR'+ str(parameters.false_positive_rate).replace('.','') + '_FNR'+ str(parameters.false_negative_rate).replace('.','')
#     if(os.path.exists(d)) == False:
#       os.mkdir(d)
#       
#     likelihood_at_k, all_results = ppinetsim.estimate_likelihood(parameters, num_simulations_per_generator=nsg)
#     likelihood_at_k.to_csv(d+'/likelihood_'+parameters.test_method + '_'+ 'accTh' + str(parameters.acceptance_threshold).replace('.','') + '_FPR'+ str(parameters.false_positive_rate).replace('.','') + '_FNR'+ str(parameters.false_negative_rate).replace('.','') +'.csv', index = False)
#     #save results in json
#     results4json=[]
#     for result in all_results:
#       temp=list(result)
#       temp[2]=[list(temp[2][0]),list(temp[2][1])]
#       results4json.append(tuple(temp))
#       json.dump(results4json, codecs.open(d+'/all_results.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


##à#### version with Parallel #####

def simulation_forParallel(m,f,nsg):
  """Runs the simulation for several combinations of parameters.

    Parameters
    ----------
    m = method
    f = file of parameters
    nsg = number of simulations for generator
    
    Returns
    -------
    save outputs of estimate_likelihood in csv and json files
    """
  parameters = ppinetsim.Parameters('parameter_settings/all_param_combinations/'+m+'/'+f)
  print(f)  
  if os.path.exists('output_results/'+str(parameters.test_method)) == False:
    os.mkdir('output_results/'+str(parameters.test_method))
    
  d = 'output_results/'+str(parameters.test_method)+'/accTh' + str(parameters.acceptance_threshold).replace('.','') + '_FPR'+ str(parameters.false_positive_rate).replace('.','') + '_FNR'+ str(parameters.false_negative_rate).replace('.','')
  if(os.path.exists(d)) == False:
    os.mkdir(d)
    
  likelihood_at_k, all_results = ppinetsim.estimate_likelihood(parameters, num_simulations_per_generator=nsg)
  likelihood_at_k.to_csv(d+'/likelihood_'+parameters.test_method + '_'+ 'accTh' + str(parameters.acceptance_threshold).replace('.','') + '_FPR'+ str(parameters.false_positive_rate).replace('.','') + '_FNR'+ str(parameters.false_negative_rate).replace('.','') +'.csv', index = False)
  #save results in json
  results4json=[]
  for result in all_results:
    temp=list(result)
    temp[2]=[list(temp[2][0]),list(temp[2][1])]
    results4json.append(tuple(temp))
  json.dump(results4json, codecs.open(d+'/all_results_'+parameters.test_method + '_'+ 'accTh' + str(parameters.acceptance_threshold).replace('.','') + '_FPR'+ str(parameters.false_positive_rate).replace('.','') + '_FNR'+ str(parameters.false_negative_rate).replace('.','')+'.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

#------------------------------------------

if os.path.exists('output_results') == False:
  os.mkdir('output_results')

method = ['AP-MS','Y2H']
nsg = 50
jobs = 8

method = ['AP-MS']
start_time = time.time()
for m in method:
  dir_parameters = 'parameter_settings/all_param_combinations/'+ m +'/'
  print(m)
  files = os.listdir(dir_parameters)
  files = files[24:32]
  #files = ['params_Y2H_accTh00_FPR005_FNR00.json','params_Y2H_accTh00_FPR00_FNR00.json']
  print(files)
  Parallel(n_jobs = jobs)(delayed(simulation_forParallel)(m,f,nsg) for f in files)
print(time.time() - start_time)




