"""
This is example code for running the MetropolisHastingsV2 sampler which can
handle lists of Tensors. It is currently setup to do and n x n and m x m matrix.
Set dim1 and dim2 for n and m respecitvely.
"""

import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from math import pi

#Import a sampler
from Samplers.metropolisHastingsV2 import MetropolisHastings


dtype = tf.float32

dim1=2
dim2=2
parallelSamples = 1 #Number of parallel chains
    

#Gaussian distributions
def gaussian(x, mu, sigma):
    a=tf.math.log(tf.cast((1/(sigma*tf.sqrt(2*pi))), dtype))
    b= tf.cast(-0.5*((x-mu)/sigma)**2, dtype)
    return(a+b)

#Donut distribution
def p(state):
    state1 = tf.reshape(state[0], [state[0].shape[0], dim1**2])
    state2 = tf.reshape(state[1], [state[0].shape[0], dim2**2])
    
    p = tf.math.reduce_sum(gaussian(state1,0,1),axis=1)
    p = p + tf.math.reduce_sum(gaussian(state2,3,1),axis=1)
    return(p)

def main():    
    
    start = time.time()
    
    dimensions=2 #2d distribtuion
    alpha = 2.5 #max change in a dimension
    samples = 1000 #Saved samples along each chain
    burnin = 1000 #Number of samples along each chain before saving
    target=0.25 #Target acceptance rate
    
    
    
    startingVals = [tf.cast(np.random.normal(0,1,(parallelSamples,dim1,dim1)), tf.float32),
                    tf.cast(np.random.normal(0,1,(parallelSamples,dim2,dim2)), tf.float32)]
    
    #Declare the sampler
    mhSampler = MetropolisHastings(dimensions, p, alpha, 
                                   parallelSamples=parallelSamples,
                                   dtype=dtype, target=target)
    
    #Run the sampler
    mhSampler.run_sampler(startingVals,burnin, samples, parallelSamples)
    
    #Extract the results
    sampledVals, acceptanceRate = mhSampler.extract_states()
    
    #Display basic results
    print("Acceptance rate", acceptanceRate)
    totalTime = time.time()-start
    print("Time elapsed:", totalTime, "seconds")
    totalSamples = (samples+burnin)*parallelSamples
    print("Time elapsed per sample:", (1000000)*totalTime/(totalSamples),
          "micro seconds")
    
    #print(sampledVals)
    for x in sampledVals:
        print(x.shape)
    for x in range(dim1):
        for y in range(dim1):
            plt.figure()
            plt.hist(sampledVals[0][:,:,x,y].reshape(-1))
            plt.show()
    for x in range(dim2):
        for y in range(dim2):
            plt.figure()
            plt.hist(sampledVals[1][:,:,x,y].reshape(-1))
            plt.show()
    
    


if(__name__=="__main__"):
    main()
