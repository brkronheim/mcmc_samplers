import tensorflow as tf

from sampler import Sampler

class MetropolisHastings(Sampler):
    """
    An implementation of the Metropolis-Hastings MCMC sampler. This sampler
    works by proposing a new state which differs in each dimension by a random
    amount in the range (-alpha, alpha). The new state is always accepted if
    it has a higher probability than the previous state, and accepted with
    probability p_new/p_current if p_new<p_current.
    """
    
    def __init__(self, dimensions, log_likelihood_fn, alpha):
        """
        The constructor for the Metropolis-Hastings sampler

        Parameters
        ----------
        dimensions : the number of dimensions of the sampled distribution
        log_likelihood_fn : a function accepting a state from the distribution
            and returning the natural logarithm of the probability of that
            state. This probability need not be normalized.
        alpha : a parameter determining the maximum change in a state 
            component during a single sampling step
            

        Returns
        -------
        None.

        """
        self.dimensions=dimensions
        self.log_likelihood_fn = log_likelihood_fn
        self.alpha = alpha
        self.sampledVals = []
        self.acceptance=0   


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
        
        #Use float32 for ease of use with GPUs
        currentState = tf.cast(initial_state, tf.float32) 
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
        sampledVals = tf.TensorArray(tf.float32, size=self.samples,
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
        A function which performs one step of the Metropolis-Hastings sampler

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        currentProb : the current probabilities of the states

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedProb : the updated probabilities of the states
        accepted : 1 if the new state was accepted, 0 otherwise for each chain
        
        """
        newState = self.propose_states(currentState)
        updatedState, updatedProb, accepted = self.accept_reject(currentState,
                                                                 currentProb,
                                                                 newState)
        
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
        step = self.alpha*tf.random.uniform(currentState.shape, minval=-1,
                                            maxval=1, dtype=tf.float32)
        newState=tf.add(currentState, step)
        return(newState)

   
    def accept_reject(self, currentState, currentProb, newState):
        """
        

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        currentProb : the current probabilities of the states
        newState : the next proposed state for all the parallel chains

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedProb : the updated probabilities of the states
        accepted : 1 if the new state was accepted, 0 otherwise for each chain

        """
        newProb = self.log_likelihood_fn(newState)
        
        randomAccept = tf.random.uniform(newProb.shape, dtype=tf.float32)
        acceptCriteria = randomAccept<tf.math.exp(newProb-currentProb)
        
        accept = tf.where(acceptCriteria,1, 0)
        updatedProb = tf.where(acceptCriteria, newProb, currentProb)
        updatedState = tf.where(acceptCriteria, newState, currentState)
        
        return(updatedState, updatedProb, accept)
    