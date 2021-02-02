import numpy as np

class Sampler(object):
    """
    Markov Chain Monte Carlo (MCMC) Parent class. This class provides the 
    general structure for the MCMC samplers.    
    """
    
    def __init__(self, dimensions, log_likelihood_fn):
        """
        The constructor accepts general information about the sampling task
        and sampler such as the number of dimensions of the target space and
        a log likelihood function for it. This will likely be overwritten for
        actual sampler implementations if there are sampler specific
        parameters.
        
        Parameters
        ----------
        dimensions : the number of dimensions of the sampled distribution
        log_likelihood_fn : a function accepting a state from the distribution
            and returning the natural logarithm of the probability of that
            state. This probability need not be normalized.     

        Returns
        -------
        None.

        """
        
        self.dimensions=dimensions
        self.log_likelihood_fn = log_likelihood_fn
        
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
    
    def run_sampler_inner(self):
        """
        The run_sampler_inner function is meant to be autographed with a 
        tf.function in order to maximize sampling speed. It should not directly
        modify class variables, instead returning any variables which need
        changed to run_sampler.
        """
        pass
    
    def propose_states(self):
        """
        The propose_states function is intended to do exactly what the name
        implies, supply the next possible location for the sampler.
        """
        pass
    
    def accept_reject(self):
        """
        The accept_reject function decides whether a proposed_state should be
        accepted or not.
        """
        pass
    
    def extract_states(self):
        """
        The extract_states function returns the results of the sampler,
        specifically the states and the acceptanceRate, which is the fraction
        of the time the new state was accepted. Unless the sampler returns
        more information than this, this function likely will not need
        to be overwritten.

        Parameters
        ----------
        None

        Returns
        -------
        states : The sampled states from the sampler.
        acceptanceRate: The proportion of the time a new state was accepted

        """

        states = np.array(self.sampledVals)
        totalSamples = self.samples*self.parallel_chains
        acceptanceRate = np.array(self.acceptance/(totalSamples))
        return(states, acceptanceRate)
    