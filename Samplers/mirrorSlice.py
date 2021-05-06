import tensorflow as tf

from Samplers.sampler import Sampler

class MirrorSlice(Sampler):
    """
    An implementation of the Mirror Slice Sampling MCMC sampler. This sampler
    works by picking a random direction, moving a set distance in that
    direction, then reflecting off the gradient of the distribution if it has
    exited the slice. The slice is all regions of the distribution with a
    probability greater than k, where k is between 0 and the proability of 
    the starting location. A state is accepted if it is inside the slice at 
    the end.
    """
    
    def __init__(self, dimensions, log_likelihood_fn, step_size_min,
                 step_size_max, mirror_steps, dtype=tf.float32):
        """
        The constructor for the Mirror Slice sampler

        Parameters
        ----------
        dimensions : the number of dimensions of the sampled distribution
        log_likelihood_fn : a function accepting a state from the distribution
            and returning the natural logarithm of the probability of that
            state. This probability need not be normalized.
        step_size_min: the minimum value for the distance moved with each step
        step_size_max: the maximum value for the distance moved with each step
        mirror_steps: the number of mirror steps in one sample
            

        Returns
        -------
        None.

        """
        self.dtype = dtype
        self.dimensions=dimensions
        self.log_likelihood_fn = log_likelihood_fn
        self.sampledVals = []
        self.acceptance=0  
        self.step_size_min = tf.cast(step_size_min, self.dtype)
        self.step_size_max = tf.cast(step_size_max, self.dtype)
        self.mirror_steps = mirror_steps
        


    @tf.function   
    def run_sampler_inner(self, initial_state):
        """
        The inner sampling step for the Mirror Slice sampler

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
        
        currentState = tf.cast(initial_state, self.dtype) 
        currentProb = self.log_likelihood_fn(currentState)
        
        
        #run sampler for the number of burn_in steps
        i = tf.constant(0)
        samples = self.burn_in
        condition = lambda i, currentState, currentProb: tf.less(i, samples)
        
        #Body of while loop for burn_in, no states or acceptance info kept
        def body(i, currentState, currentProb):
            currentState, currentProb, accept = self.one_step(currentState,
                                                              currentProb)
            
            return([tf.add(i, 1), currentState, currentProb])

        #tf.while_loop to speed up sampling
        i, currentState, currentProb = tf.while_loop(condition, body, 
                                                     [i, currentState, 
                                                      currentProb])
        
        #tensorArray of set size to store samples
        sampledVals = tf.TensorArray(self.dtype, size=self.samples,
                                     dynamic_size=False)
        
        #run sampler for the number of sampling steps
        samples += self.samples
        def condition(i, currentState, currentProb, sampledVals, acceptance):
            return(tf.less(i, samples))
        
        #Body of while loop for sampling, states and acceptance info kept
        def body(i, currentState, currentProb, sampledVals, acceptance):
            
            currentState, currentProb, accept = self.one_step(currentState,
                                                              currentProb)
            acceptance+=tf.reduce_sum(accept)
            sampledVals= sampledVals.write(i-self.burn_in, currentState)
            
            
            return([tf.add(i, 1), currentState, currentProb, sampledVals,
                    acceptance])

        #tf.while_loop to speed up sampling
        i, currentState, currentProb, sampledVals,acceptance = \
            tf.while_loop(condition, body, [i, currentState, currentProb,
                                            sampledVals, acceptance])
        
        #trun sampledVals into a normal Tensor
        sampledVals = sampledVals.stack()                                                                                    
        
        return(sampledVals, acceptance)

        
        
    def one_step(self, currentState, currentProb):
        """
        A function which performs one step of the Mirror Slice sampler

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        currentProb : the current probabilities of the states

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedProb : the updated probabilities of the states
        accepted : acceptance probability for each chain
        
        """
        newState, newProb, slice_prob = self.propose_states(currentState)
        updatedState, updatedProb, accepted = self.accept_reject(currentState,
                                                                 newState, slice_prob, newProb)
    
        
        
        return(updatedState, updatedProb, accepted)

        
        
    def propose_states(self, currentState):
        """
        

        Parameters
        ----------
        currentState : the current state of the sampler for each chain

        Returns
        -------
        newState : the proposed new state of the sampler for each chain

        """
        i = tf.constant(0)
        self.step_size=tf.random.uniform((),self.step_size_min, self.step_size_max, dtype=self.dtype)
        currentMomentum = tf.random.normal(currentState.shape, dtype=self.dtype)
        currentProb = self.log_likelihood_fn(currentState)
        slice_prob = tf.math.log(tf.random.uniform([1], minval=0, maxval = tf.exp(currentProb), dtype=self.dtype))
        
        
        def one_mirror_step(i, currentState, currentMomentum, currentProb):
            newState = currentState + currentMomentum*self.step_size
            prob = None
            with tf.GradientTape() as g:
              g.watch(newState)
              prob = self.log_likelihood_fn(newState)
              
            grad = g.gradient(prob, newState)
            newMomentum = tf.where(prob>slice_prob, currentMomentum, currentMomentum-2*grad*tf.reduce_sum(currentMomentum*grad)/tf.reduce_sum(grad*grad))
            
            return([tf.add(i, 1), newState, newMomentum, prob])
        
        def condition(i, currentState, currentMomentum, currentProb):
            return(tf.less(i, self.mirror_steps))
        
        i, newState, currentMomentum, currentProb = tf.while_loop(condition, one_mirror_step, [i, currentState, currentMomentum, currentProb])
        
        
        return(newState, currentProb, slice_prob)

   
    def accept_reject(self, currentState, newState, slice_prob, new_prob):
        """
        

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        newState : the next proposed state for all the parallel chains
        slice_prob : lowest probability of slice
        new_prob : probabiltiy of new state

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedProb : the updated probabilities of the states
        accepted : 1 if the new state was accepted, 0 otherwise for each chain

        """
        acceptCriteria = new_prob>=slice_prob
        
        accept = tf.where(acceptCriteria,1, 0)
        updatedState = tf.where(acceptCriteria, newState, currentState)
        
        return(updatedState, new_prob, accept)
    