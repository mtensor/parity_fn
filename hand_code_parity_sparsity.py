import numpy as np

#define function, output a tuple of W list and b list
def hand_code_optimal_parity_fn_sparsity_pattern(n,weightscale):
    logn = int(np.ceil(np.log2(n)))
    W = []
    b = []
    for s in range(logn):
        size = 2 ** (logn - s)
        
        #make layer1   
        layer1 = np.zeros((size,size),dtype=np.float32)
        for i in range(size/2):
                layer1[2*i,2*i] = 1
                layer1[2*i+1,2*i+1] = -1
                layer1[2*i+1,2*i] = -1
                layer1[2*i,2*i+1] = 1     
        #using "xavier initialization"
        std_dev = weightscale * np.sqrt(2. / (size + size))
        A = np.random.normal(scale=std_dev,size=(size,size))
        layer1 = layer1 + A.astype(np.float32) * (layer1 !=0)
        W.append(layer1)
        
        #make bias1
        bias1 = np.zeros((size,1),dtype=np.float32)
        bias1[np.array(range(size)) % 2 == 0] = 0.5
        bias1[np.array(range(size)) % 2 == 1] = -1.5
              
        std_dev = weightscale * np.sqrt(2. / (size + size))
        A = np.random.normal(scale=std_dev,size=(size,1))
        bias1 = bias1 + A.astype(np.float32) * (bias1 !=0)
        b.append(bias1)
        
        #make layer2
        layer2 = np.zeros((size/2,size),dtype=np.float32)
        for i in range(size/2):
            layer2[i,2*i] = 1
            layer2[i,2*i+1] = 1 
        #xavier initialization          
        std_dev = weightscale * np.sqrt(2. / (size/2 + size))
        A = np.random.normal(scale=std_dev,size=(size/2,size))
        layer2 = layer2 + A.astype(np.float32) * (layer2 !=0)
        W.append(layer2)        
        
        bias2 = np.ones((size/2,1),dtype=np.float32)*1.5
        std_dev = weightscale * np.sqrt(2. / (size/2 + size))
        A = np.random.normal(scale=std_dev,size=(size/2,1))
        bias2 = bias2 + A.astype(np.float32) * (bias2 != 0)
        b.append(bias2)
    return (W,b)
 


def hand_code_identity_fn(n,std_dev):
    raise('hand_code_identity_fn not yet ready')
    logn = int(np.ceil(np.log2(n)))
    W = []
    b = []
    for i in range(logn):
        size = 2 ** (logn - i)
        #make ID matrix with size
        layer = np.eye(size/2,size,dtype=np.float32)
        A = np.random.normal(scale=std_dev,size=(size/2,size))
        layer = layer + A.astype(np.float32)
        W.append(layer)
        bias = np.zeros(size/2,1,dtype=np.float32)
        A = np.random.normal(scale=std_dev,size=(size/2,1))
        bias = bias + A.astype(np.float32)
        b.append(bias)
    return (W, b)
