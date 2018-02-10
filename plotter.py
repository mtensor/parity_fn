#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:05:09 2017

@author: Maxwell

Run averaging code:
    
    Designed to work with the odessey code "launch_fouriernetwork."
    
    Assuming all the runs in an experiment have been done with the same hyperparameters, 
    this code will average the desired parameters
"""

import glob
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('-expt', type=int)
parser.add_argument('-size', type=int)

settings = parser.parse_args(); 

experiment_num = settings.expt
size = settings.size

basepath = "/n/home09/mnye/parity_fn/results/expt%d/" % experiment_num
directory_path = basepath + 'data/'
plotpath = basepath + 'plots/'
print 'directory_path:', directory_path

fun_loss_list = []
weightscale_list = []
for res_num in glob.glob(directory_path + '*.npz'):
    #try: 
    variables = np.load(res_num)
    run_params = variables['params'][0]
    print "run_params[0]", run_params[0]
    print "size:", run_params.size    
    if run_params.size == size:
        fun_loss_list.append(variables['fnloss'])
        weightscale_list.append(run_params.weightscale)

#    except IOError:
#        print("there exists a trial which is not complete")

        #Whatever man    
assert len(fun_loss_list) == len(weightscale_list)

fig = plt.figure()
fig, ax = plt.subplots()
print "weightscale", weightscale_list
print "fun loss list", fun_loss_list
plt.plot(weightscale_list,fun_loss_list)
ax.set(title='Parity function convergence',
       xlabel='Weight initialization noise scale',
       ylabel='Final network error')

#fig.savefig(plotpath + "paritysize%d.png" % size, dpi=200)
fig.savefig('parityexpt%dsize%d.png' %(experiment_num, size), dpi = 200)

 
