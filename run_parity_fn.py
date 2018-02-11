from __future__ import print_function
from __future__ import division
import numpy as np
import sys
import argparse
#max's
import tensorflow as tf
from hand_code_parity_fn import hand_code_optimal_parity_fn
from hand_code_parity_fn import hand_code_identity_fn
from hand_code_parity_sparsity import hand_code_optimal_parity_fn_sparsity_pattern


parser = argparse.ArgumentParser()

parser.add_argument('-paramfile', type=argparse.FileType('r')) #keep
parser.add_argument('-line', type=int)
parser.add_argument('-rseed', type=int, default=1) #Keep
parser.add_argument('-rseed_offset', type=int, default=3) #Keep

parser.add_argument('-epochs',     type=int, default=5000) #Keep
parser.add_argument('-weightscale', type=float, default=0.2) #Keep
parser.add_argument('-beta', type=float, default=0.0001) #0.0001
parser.add_argument('-optimizer', type=float, default=0.0001)
parser.add_argument('-size', type=int, default=16)
parser.add_argument('-runtoconv', action='store_true')
parser.add_argument('-batch_size', type=int, default = 1000)
parser.add_argument('-hidden_width_multiplier', type=float, default = 1.0)

parser.add_argument('-savefile', type=argparse.FileType('w'))
parser.add_argument('-showplot', action='store_true')
parser.add_argument('-saveplot', action='store_true')
parser.add_argument('-verbose', action='store_true')
parser.add_argument('-sparse', action='store_true')

settings = parser.parse_args(); 
                            
# Read in parameters from correct line of file
if settings.paramfile is not None:
    for l, line in enumerate(settings.paramfile):
        if l == settings.line:
            settings = parser.parse_args(line.split())
            break
            
if settings.showplot or settings.saveplot:
    import matplotlib
    
    if not settings.showplot:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

np.random.seed(settings.rseed_offset)


# initial conditions
n = settings.size
logn = int(np.ceil(np.log2(n)))
train_time = settings.epochs
optimizer_parameter = settings.optimizer #it sometimes converges at .001
beta = settings.beta# 0.01 #needs to be dynamically adjusted???
loss_print_period = int(np.ceil(train_time/100.))
traintoconv = settings.runtoconv
inv_temp = 10#TODO
batch_size = settings.batch_size
print("layerwise L1 is on")


#weight initialization
 #SHOULD BE DIFFERENT FOR EACH LAYER

######Initialize near optimal solution######
if settings.sparse:
    (W_init,bias_init) = hand_code_optimal_parity_fn_sparsity_pattern(n,settings.weightscale)
    print("starting near optimial solution, maintaining sparsity pattern")
else:
    (W_init,bias_init) = hand_code_optimal_parity_fn(n,settings.weightscale)
    print("starting near optimial solution")


W = [tf.Variable(W_init[i]) for i in range(len(W_init))]
bias = [tf.Variable(bias_init[i]) for i in range(len(bias_init))]

"""
######Initialize with large hiddden layers######
#TODO
"""

"""
######Initialize with identity matrix######
(W,bias) = hand_code_identity_fn(n, settings.weightscale)
print("initialized with noisy identity matrix")
"""

# network layers
input_vec = tf.placeholder(tf.float32, shape=[n,None])

hidden = [input_vec]
for i in range(len(W)):
    #TODO use sigmoids with biases
    hidden.append(tf.sigmoid(inv_temp*(tf.matmul(W[i],hidden[-1]) - bias[i])))
output = hidden[-1]
#TODO correct shape for output layer
parity_output = tf.placeholder(tf.float32, shape=[1,None])

#functions for use
def l0norm(W,b):
    norm0 = 0
    for i in range(len(W)):
        ones = np.ones(W[i].shape)
        norm0 = norm0 + np.sum(ones[W[i] != 0])
    for i in range(len(b)):
        ones = np.ones(W[i].shape)
        norm0 = norm0 + np.sum(ones[W[i] != 0])      
    return norm0
    
def l_1_norm(W,b):
    l1 = 0
    for i in range(len(W)):
        l1 = l1 + np.sum(abs(W[i]))
    for i in range(len(b)):
        l1 = l1 + np.sum(abs(b[i]))    
    return l1

def mat_l_1_norm(W):
    l1 = 0
    for i in range(len(W)):
        l1 = l1 + np.sum(abs(W[i]))   
    return l1

def rectify(W,b,cutoff):
    W_rect = W
    for i in range(len(W)):
        W_rect[i][np.abs(W[i]) < cutoff] = 0
    b_rect = b
    for i in range(len(b)):
        b_rect[i][np.abs(b[i]) < cutoff] = 0        
    return (W_rect,b_rect)
 
def eval_network(input_mat,W,b):
    #TODO
    ft = [input_mat]
    for l in range(len(W)):
        ft.append(np.matmul(W[l],ft[-1]))
    ft = ft[-1]
    return ft
    
def calc_error(input_mat,output,W):
    #TODO
    #Uses continuous definition of error 
    fn = eval_network(input_mat, W)
    diff = fn - np.sum(input_mat, axis=0, keepdims=True) % 2
    rect_error = sum(sum(np.square(diff)))
    return rect_error

    #training data

def training_data(n, batch_size):
    input_train = np.random.randint(2, size=(n,batch_size))
    output_train = np.sum(input_train, axis=0, keepdims=True) % 2
    return (input_train, output_train)

    
def key_cutoff_finder(W, max_cutoff):
    specificity = 1000
    key_cutoff = 0.
    key_cutoff_error = calc_error(np.identity(W[0].shape[0]), W)
    cutoff_list = np.arange(0.,max_cutoff, specificity)
    for cutoff in cutoff_list:
        cutoff_error = calc_error(np.identity(W[0].shape[0]), rectify(W, cutoff))
        if cutoff_error <= key_cutoff_error:
            key_cutoff = cutoff
            key_cutoff_error = cutoff_error
    return key_cutoff

    
# loss
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=1.0, scope=None)
layer_penalty = []
for i in range(len(W)):
    layer_penalty.append(tf.square(tf.contrib.layers.apply_regularization(
            l1_regularizer, weights_list=[W[i]])))
    layer_penalty.append(tf.square(tf.contrib.layers.apply_regularization(
            l1_regularizer, weights_list=[bias[i]])))
regularization_penalty = tf.add_n(layer_penalty)
fn_loss = tf.reduce_sum(tf.square(output - parity_output))
#The important line: 
regularized_loss = fn_loss + beta * regularization_penalty

# optimizer
optimizer = tf.train.AdamOptimizer(optimizer_parameter)
train = optimizer.minimize(regularized_loss)

    

#part two stuff



# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


print("n: %s" %n)
print("initial total weight variance scale: %s" %settings.weightscale)
#calculate optimal l1 norm
(W_opt,b_opt) = hand_code_optimal_parity_fn(n,0)
layerwise_opt_norm_W = [mat_l_1_norm(W_opt[i]) for i in range(len(W_opt))]
layerwise_opt_norm_b = [mat_l_1_norm(b_opt[i]) for i in range(len(b_opt))]
optimal_L1 = np.sum(np.square(layerwise_opt_norm_W+layerwise_opt_norm_b))*beta
print("optimal L1 norm: %s" %(optimal_L1))


#prints for first loop:
print("Using Adam Optimizer")
print("batch size: %s" %batch_size)
print("beta value: %s" %beta)
print("optimizer value: %s" %optimizer_parameter)
print("hidden width multiplier: %s" %settings.hidden_width_multiplier)

reglossvec = []
fnlossvec = []
loss_trigger = False
min_fn_loss = 0.001*n
i = 0
#loop 1
while (i < train_time) and not loss_trigger:
    
    #i just need to say what train is
    #fix batch training situation
    input_train, output_train = training_data(n, batch_size)
    d = {input_vec:input_train,parity_output:output_train}
    reg_loss_val,fn_loss_val, _ = sess.run([regularized_loss,fn_loss, train], feed_dict=d)
    
    if i%loss_print_period == 0:
        print("step %s, function loss: %s, regularized loss: %s" 
              %(i,fn_loss_val,reg_loss_val))
        sys.stdout.flush()
    
    reglossvec.append(reg_loss_val)
    fnlossvec.append(fn_loss_val)
            
    i += 1
    if fn_loss_val < 0:#complex_n:
        loss_trigger = True
    

###############################################################################
############################ after training loop ##############################
if (reg_loss_val > optimal_L1):
    print("did not train to convergence")
    
Wcurr,bcurr = sess.run([W,bias])

(W_opt,b_opt) = hand_code_optimal_parity_fn(n,0)
optimal_L0 = l0norm(W_opt,b_opt)
nlogn = float(n * logn)
optimal_scale_factor = optimal_L0 / nlogn

print("\t L_0 norm: %g (hand-coded value is %g) "%(l0norm(Wcurr,bcurr), optimal_L0))

print("\t l1 norm: %g (hand-coded value is %g)"%(l_1_norm(Wcurr,bcurr), l_1_norm(W_opt,b_opt)))


#key cutoff finder:
val_size = 5000
val_in, val_out = training_data(n,val_size)


"""
cutoff_list = [1., 2., 5., 10., 20., 50., 100.] #need to be floats
rect_errors = []
l0_norms = []
scaling_factors = []

(W_opt,b_opt) = hand_code_optimal_parity_fn(n,0)
optimal_L0 = l0norm(W_opt,b_opt)
nlogn = float(n * logn)
optimal_scale_factor = optimal_L0 / nlogn

for index in range(len(cutoff_list)):
    cutoff_factor = cutoff_list[index]
    
    Wcurr = sess.run(W)  
    
    #find cutoffval
    cutoff_val = 0.5
    W_rect = rectify(Wcurr,cutoff_val)
    print("Cutoff factor: %s" %cutoff_factor)
    
    #calculate error
    ft_in = np.identity(n)
    
    rect_error = calc_error(ft_in, W_rect)
    rect_errors.append(rect_error)
    print("\t Function error of rectified network: %s" %rect_error)
    
    #calculate L0 norm 
    l0_norm = l0norm(W_rect)
    l0_norms.append(l0_norm)
    print("\t L_0 norm: %g (hand-coded value is %g) "%(l0_norm, optimal_L0))
    
    #calculate scaling factor
    nlogn = float(n * logn)
    scaling_factor = l0_norm / nlogn
    scaling_factors.append(scaling_factor)
    print("\t Complexity scaling factor: %g (hand-coded value is %g)" %(scaling_factor, optimal_scale_factor))

Wcurr = sess.run(W)
max_cutoff = 0.5
key_cutoff = key_cutoff_finder(Wcurr, max_cutoff)

key_cutoff_factor = max_cutoff/key_cutoff
print("Key_cutoff_factor: %g" %(key_cutoff_factor))
"""

if settings.savefile:
    np.savez(settings.savefile, W=Wcurr, params=[settings], fnloss=fn_loss_val, reglossvec=reglossvec, fnlossvec=fnlossvec)


#deal with this later 


'''
unitwise_loss = tf.square(output - ft_output)
unit_loss = sess.run(unitwise_loss,{input_vec:np.identity(n),ft_output:fourier_trans(np.identity(n))})
import matplotlib.pyplot as plt
plt.imshow(unit_loss), plt.colorbar()

unitwise_output = sess.run(output,{input_vec:input_train,ft_output:output_train})
'''


"""   
if settings.savefile:
    np.savez(settings.savefile, train=train_err, test=test_err, params=[settings])

if settings.showplot or settings.saveplot:
    epoch = np.linspace(0, len(history.history['loss']), len(history.history['loss']))

    fig, ax = plt.subplots(1)
    line1, = ax.plot(epoch, history.history['loss'], linewidth=2,label='Train loss')
    line2, = ax.plot(epoch, history.history['val_loss'], linewidth=2, label='Test loss')
    ax.legend(loc='upper center')

    if settings.showplot:
        plt.show()
    elif settings.saveplot:
        fig.savefig('rand_relu_training_dynamics.pdf', bbox_inches='tight')
"""


