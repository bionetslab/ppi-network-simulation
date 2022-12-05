import numpy as np
import scipy.optimize as op
import scipy.special as sp
import time
import rpy2
import rpy2.robjects
import os
import ppinetsim
import ppinetsim.utils as utils
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


######################### poweRlaw package ############################

def callPoweRlaw(distributionYs, simNum, t):
    v = rpy2.robjects.IntVector(distributionYs)
    rpy2.robjects.r('''library("poweRlaw")
        saveVec <- function (v) {
            assign("v", v, envir = .GlobalEnv)
        }''')
    rpy2.robjects.r['saveVec'](v)
    run =  rpy2.robjects.r('''
        pl_v <- displ$new(v)
        est <- estimate_xmin(pl_v)
        if(is.na(est$xmin) || is.infinite(est$gof)){
          c(NA, NA, NA, NA, NA)
        }else{
        pl_v$setXmin(est)
        bs_p <- bootstrap_p(pl_v, no_of_sims = ''' + str(simNum) + ''', threads = ''' + str(t)+ ''', seed = 1)
        plot(pl_v)
        lines(pl_v, col = 'green')
        # category
        plCat <- 0
        if(bs_p$p > 0.1){
          plCat <- 2
          meet <- 0
          for (tail in bs_p$bootstrap$ntail) {
            if(tail>50){
              meet <- meet+1
            }
          }
          if((meet/length(bs_p$bootstrap$ntail)) > 0.75){
            plCat <- 3
            meet <- 0
            for (alpha in bs_p$bootstrap$pars) {
              if(alpha>2 && alpha<3){
                meet <- meet+1
              }
            }
            if((meet/length(bs_p$bootstrap$pars)) > 0.5){
              plCat <- 4
              if((meet/length(bs_p$bootstrap$pars)) > 0.9){
                plCat <- 5
              }
            }
          }
        }
        c(bs_p$p, est$xmin, est$pars, est$ntail,plCat)
        }
        ''')
    return [run[0], run[1], run[2], run[3], run[4]]

def calculate_final_degree_ground_truth(f,adj_ground_truth,parameters,pl_num,t,dir_output,i):
  
  degree_ground_truth = utils.degree_distribution(adj_ground_truth)
  degree_ground_truth = degree_ground_truth[degree_ground_truth != 0]
  np.savetxt(dir_output + 'degree_dist_file/'+'degree_ground_truth_'+str(i)+'_'+str(f.replace('.json',''))+'.txt', degree_ground_truth, delimiter='\n')
  r = callPoweRlaw(degree_ground_truth, pl_num,t)
  r.append('groundtruth_'+f)
  final_vector = [parameters.num_proteins, parameters.num_ppis_ground_truth, parameters.false_positive_rate, parameters.false_negative_rate, parameters.generator, parameters.biased, parameters.baseline_degree, parameters.test_method, parameters.num_baits, parameters.star_size_constant, parameters.num_preys, parameters.matrix_size_constant, parameters.acceptance_threshold, parameters.num_studies, parameters.max_num_ppis_observed, utils.num_edges(adj_ground_truth) / 2, len(degree_ground_truth), i]
  for j in r:
    final_vector.append(j)
  return final_vector


def run_iterations(files,n,dir_parameters,pl_num,t,dir_output,name_file):
  """Runs the simulation n times.

    Parameters
    ----------
    files = names of the files which include the parameters
    n = number of iterations (= number of observed networks)
    dir_parameters = directory where the parameters files are located
    pl_num = number of iterations used in bootstrappping procedure (poweRlaw package; default = 100)
    t = number of cores used to run the bootstrapping procedure
    dir_output = directory of the outputs
    name_file = name of the output file
    
    Returns
    -------
    csv file saved in the dir_output.
    """
  final_table = []
  for f in files:
    #print(f)
    parameters = ppinetsim.Parameters(dir_parameters+f)
  
    for i in range(n):
      degree_distributions, num_ppis, adj_ground_truth = ppinetsim.run_simulation(parameters, verbose=True)
      degree_distribution = degree_distributions[-1:]
      len(degree_distribution[0]) # == num_proteins
      # remove zero
      degree_distribution = degree_distribution[degree_distribution != 0]
      len(degree_distribution)
  
      r = callPoweRlaw(degree_distribution,pl_num,t)
      r.append(f)
  
      final_vector = [parameters.num_proteins, parameters.num_ppis_ground_truth, parameters.false_positive_rate, parameters.false_negative_rate, parameters.generator, parameters.biased, parameters.baseline_degree, parameters.test_method, parameters.num_baits, parameters.num_preys, parameters.acceptance_threshold, parameters.num_studies, parameters.max_num_ppis_observed, num_ppis[-1] / 2, len(degree_distribution)]
      for j in r:
        final_vector.append(j)
      final_table.append(final_vector)
      final_vector_ground = calculate_final_degree_ground_truth(f,adj_ground_truth,parameters,pl_num,t)
      final_table.append(final_vector_ground)
  
  df = pd.DataFrame(final_table, columns = ['num_proteins','num_ppis_ground_truth','false_positive_rate','false_negative_rate','generator','biased','baseline_degree','test_method','star_size','matrix_size','acceptance_threshold','max_num_test','max_num_ppis_observed','num_ppis','num_proteins_after_simul','pvalue','xmin','pars','ntail','plCat','type'])
  df.to_csv(dir_output + name_file +'.csv', index = False)

#################
# function to make plot
# def make_plot(degree_distribution,dir_output,i,f):
#   
#   freqs = np.asarray(np.unique(degree_distribution, return_counts=True))
#   plt.loglog(freqs[0,],freqs[1,],'go-')
#   plt.xlabel('Degree')
#   plt.ylabel('Frequency')
#   plt.savefig(dir_output + 'plot_observed_python/' + 'plot_'+ str(f.replace('.json','')) + '_iter' + str(i) + '.png')
  
#########
def fun(parameters,f,pl_num,t,final_table,i,dir_output):
  
  degree_distributions, num_ppis, adj_ground_truth = ppinetsim.run_simulation(parameters, verbose=True)
  degree_distribution = degree_distributions[-1:]
  np.savetxt(dir_output + 'degree_dist_file/'+'degree_observed_'+str(i)+'_'+ str(f.replace('.json','')) +'.txt', degree_distribution[0], delimiter='\n')
  len(degree_distribution[0]) # == num_proteins
  # remove zero
  degree_distribution = degree_distribution[degree_distribution != 0]
  len(degree_distribution)
  #make_plot(degree_distribution,dir_output,i,f)
  #np.savetxt(dir_output + 'degree_dist_file/'+'degree_observed_'+str(i)+'_'+ str(f.replace('.json','')) +'.txt', degree_distribution, delimiter='\n')
  r = callPoweRlaw(degree_distribution,pl_num,t)
  r.append(f)
  final_vector = [parameters.num_proteins, parameters.num_ppis_ground_truth, parameters.false_positive_rate, parameters.false_negative_rate, parameters.generator, parameters.biased, parameters.baseline_degree, parameters.test_method, parameters.num_baits, parameters.star_size_constant, parameters.num_preys, parameters.matrix_size_constant, parameters.acceptance_threshold, parameters.num_studies, parameters.max_num_ppis_observed, num_ppis[-1] / 2, len(degree_distribution), i]
  for j in r:
    final_vector.append(j)
  final_table.append(final_vector)
  #final_vector_ground = calculate_final_degree_ground_truth(f,adj_ground_truth,parameters,pl_num,t,dir_output,i)
  #final_table.append(final_vector_ground)
  return(final_table)
  

def run_iterations_withParallel(files,n,dir_parameters,pl_num,t,dir_output,name_file,jobs):
  """Runs the simulation n times.

    Parameters
    ----------
    files = names of the files which include the parameters
    n = number of iterations (= number of observed networks)
    dir_parameters = directory where the parameters files are located
    pl_num = number of iterations used in bootstrappping procedure (poweRlaw package; default = 100)
    t = number of cores used to run the bootstrapping procedure
    dir_output = directory of the outputs
    name_file = name of the output file
    
    Returns
    -------
    csv file saved in the dir_output.
    """
  if os.path.exists(dir_output + 'degree_dist_file') == False:
    os.mkdir(dir_output + 'degree_dist_file')
  # if os.path.exists(dir_output + 'plot_observed_python') == False:
  #   os.mkdir(dir_output + 'plot_observed_python')
  data = pd.DataFrame()
  for f in files:
    print(f)
    parameters = ppinetsim.Parameters(dir_parameters+f)
    final_table = []
    final_table = Parallel(n_jobs = jobs)(delayed(fun)(parameters,f,pl_num,t,final_table,i,dir_output) for i in range(n))
    for element in final_table:
      temp=pd.DataFrame(element, columns=['num_proteins','num_ppis_ground_truth','false_positive_rate','false_negative_rate','generator','biased','baseline_degree','test_method','star_size','star_size_constant','matrix_size','matrix_size_constant','acceptance_threshold','max_num_test','max_num_ppis_observed','num_ppis','num_proteins_after_simul','n_iter','pvalue','xmin','pars','ntail','plCat','type'])
      data = pd.concat([data,temp])
  
  data.to_csv(dir_output + name_file +'.csv', index = False)
  
  
  
