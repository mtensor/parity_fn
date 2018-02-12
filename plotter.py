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
    if run_params.size == size:
        fun_loss_list.append(variables['fnloss'])
        weightscale_list.append(run_params.weightscale)

#    except IOError:
#        print("there exists a trial which is not complete")

        #Whatever man    
assert len(fun_loss_list) == len(weightscale_list)

#sort array
order = np.argsort(weightscale_list)
weightscale_list = np.array(weightscale_list)[order]
fun_loss_list = np.array(fun_loss_list)[order]

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig = plt.figure()
fig, ax = plt.subplots()
fun_loss_list = fun_loss_list / run_params.batch_size
print "weightscale", weightscale_list
print "fun loss list", fun_loss_list

plt.plot(weightscale_list,fun_loss_list, marker='o',markersize=10,linewidth=4.0)
ax.set(title='Convergence - sparsity pattern enforced',
       xlabel='Initialization noise scale',
       ylabel='Final network error')

#fig.savefig(plotpath + "paritysize%d.png" % size, dpi=200)
fig.savefig('parityexpt%dsize%d.png' %(experiment_num, size), dpi = 200)

 
