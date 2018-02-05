import numpy as np
from subprocess import call
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-expt', type=int)

settings = parser.parse_args(); 

param_fn = "params_parity_fn_%d.txt" % settings.expt
fo = open(param_fn, "w")

os.makedirs("/n/home09/mnye/parity_fn/results/expt%d/data" % settings.expt)
os.makedirs("/n/home09/mnye/parity_fn/results/expt%d/logs" % settings.expt)


#depths

weightscales = [0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2]
rseed = 2
noffsets = 4
rseed_offsets = np.linspace(0,rseed*(noffsets-1),noffsets).astype(int)
expt = settings.expt

sizes = [32]
optimizer_params = [0.0001]
L1_betas = [0.001]
batch_sizes = [1000]
hidden_width_multipliers = [1.0]

i = 1
for n in sizes:
    for bs in batch_sizes:
        for hidden_width_multiplier in hidden_width_multipliers:
            for optimizer in optimizer_params:
                for beta in L1_betas:
                    for ws in weightscales:
                        for roff in rseed_offsets:                   
                            savefile = "/n/home09/mnye/parity_fn/results/expt%d/data/res%d.npz" %(expt, i) 
                            fo.write("-rseed %d -rseed_offset %d -weightscale %g -size %d -beta %g -optimizer %g -epochs 20000 -savefile %s -batch_size %d -hidden_width_multiplier %g\n" % (rseed, roff, ws, n, beta, optimizer, savefile, bs, hidden_width_multiplier))
                            i = i+1
                            #what is lr?
                            #epoch thing may need to be cut
fo.close()

call("python run_odyssey_array.py -cmd run_parity_fn.py -expt %d -cores 8 -hours 25 -mem 24000 -partition serial_requeue -paramfile %s" % (expt,param_fn), shell=True)
