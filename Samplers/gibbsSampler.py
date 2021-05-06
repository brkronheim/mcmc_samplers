import tensorflow as tf
from Samplers.sampler import Sampler

class GibbsSampler(Sampler):
    """
    An implemenation of the Gibbs sampler. This MCMC sampler works by fixing
    all but one of the sampled dimenions and updating that dimenions according
    to its conditional distribution given the other variables. This method
    does not require an accept reject step or a step size, but it does
    require a method of exactly sampling from the conditional distribution
    """
    
    
    
    def __init__(self, dimensions, quant_fn, dtype = tf.float32):
        """
        The constructor accepts general information about the sampling task
        and sampler such as the number of dimensions of the target space and
        a log likelihood function for it. This will likely be overwritten for
        actual sampler implementations if there are sampler specific
        parameters.
        
        Parameters
        ----------
        dimensions : the number of dimensions of the sampled distribution
        quant_fn : a function accepting a state from the distribution, a 
            dimension, and a quantile which return a sample from the conditional
            distribution of the distribution given all dimenions are fixed
            except the given dimension correspodning to the quantile
        dtype : data type of data, tf.float32 is recomended

        Returns
        -------
        None.

        """
        
        self.dimensions=dimensions
        self.quant_fn = quant_fn
        self.dtype = dtype
        
    def run_sampler(self, initial_state, burn_in, samples, parallel_chains=1):
        """
        The run_sampler function accepts the specifics of the sampling run
        such as the initial state, the number of sampling steps to do, and 
        the number of parallel sampling chains being run. It then calls 
        run_sampler_inner to do the actual sampling. This function will
        not need to be overwritten in most cases.
        
        Parameters
        ----------
        initial_state : the starting point for the sampler. It should be
            structured as [[1st chain - 1st dim, 2nd chain - 1st dim , ...],
                           [1st chain - 2nd dim, 2nd chain - 2nd dim , ...],
                           ...]
        burn_in : the number of sampling steps to take prior before starting
            to save the states
        samples : the number of sampling steps to take and save post burn-in  
        parallel_chains : the number of parallel chains being run, default is 1
            

        Returns
        -------
        None.


        """
        
        self.burn_in = burn_in
        self.samples = samples
        self.parallel_chains = parallel_chains
        self.sampledVals, self.acceptance = self.run_sampler_inner(initial_state)  
    
    @tf.function   
    def run_sampler_inner(self, initial_state):
        """
        The inner sampling step for the Metropolis-Hastings sampler

        Parameters
        ----------
        initial_state : the starting point for the sampler. It should be
            structured as [[1st chain - 1st dim, 2nd chain - 1st dim , ...],
                           [1st chain - 2nd dim, 2nd chain - 2nd dim , ...],
                           ...]

        Returns
        -------
        sampledVals: the states sampled by the Metropolis-Hastings sampler
        acceptance: number of new states accepted accross all parallel
            iterations for each sampling step
        """
        
        acceptance = 0
        acceptance = tf.cast(acceptance, self.dtype)
        
        #Use float32 for ease of use with GPUs
        currentState = tf.cast(initial_state, self.dtype) 
        
        
        #run sampler for the number of burn_in steps
        i = tf.constant(0)
        samples = self.burn_in
        condition = lambda i, currentState: tf.less(i, samples)
        
        #Body of while loop for burn_in, no states or acceptance info kept
        def body(i, currentState):
            currentState, accept = self.one_step(currentState, tf.math.floormod(i, self.dimensions))
            
            return([tf.add(i, 1), currentState])

        #tf.while_loop to speed up sampling
        i, currentState = tf.while_loop(condition, body, 
                                                     [i, currentState],
                                                 shape_invariants=[i.get_shape(),tf.TensorShape([None,self.parallel_chains])])
        
        #tensorArray of set size to store samples
        sampledVals = tf.TensorArray(self.dtype, size=self.samples,
                                     dynamic_size=False)
        
        #run sampler for the number of sampling steps
        samples += self.samples
        def condition(i, currentState, sampledVals, acceptance):
            return(tf.less(i, samples))
        
        #Body of while loop for sampling, states and acceptance info kept
        def body(i, currentState, sampledVals, acceptance):
            
            currentState, accept = self.one_step(currentState, tf.math.floormod(i, self.dimensions))
            acceptance+=tf.reduce_sum(accept)
            sampledVals= sampledVals.write(i-self.burn_in, currentState)
            
            
            return([tf.add(i, 1), currentState, sampledVals,
                    acceptance])

        #tf.while_loop to speed up sampling
        i, currentState, sampledVals,acceptance = \
            tf.while_loop(condition, body, [i, currentState, 
                                            sampledVals, acceptance])
        
        #trun sampledVals into a normal Tensor
        sampledVals = sampledVals.stack()                                                                                    
        acceptance = acceptance/self.dimensions
        return(sampledVals, acceptance)

        
        
    def one_step(self, currentState, currentDim):
        """
        A function which performs one step of the Metropolis-Hastings sampler

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        currentDim : dimension being sampled

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        accepted : 1 if the new state was accepted, 0 otherwise for each chain
        
        """
        newState = self.propose_states(currentState, currentDim)
        updatedState, accepted = self.accept_reject(currentState, newState)
        
        return(updatedState, accepted)

        
        
    def propose_states(self, currentState, currentDim):
        """
        

        Parameters
        ----------
        currentState : the current state of the sampler for each chain
        currentDim : dimension being sampled

        Returns
        -------
        newState : the proposed new state of the sampler for each chain

        """
        quantile = tf.random.uniform([1,self.parallel_chains], minval=0,
                                            maxval=1, dtype=self.dtype)
        smallState = tf.concat([currentState[:currentDim],currentState[currentDim+1:]],0)
        newStateSlice=self.quant_fn(smallState, quantile, currentDim)
        newState = tf.concat([currentState[:currentDim],newStateSlice, currentState[currentDim+1:]],0)
        return(newState)
    
    def accept_reject(self, currentState, newState):
        """
        The accept_reject function decides whether a proposed_state should be
        accepted or not.
        
        States are always accepted
        """
        return(newState, tf.ones_like(currentState))
        
    