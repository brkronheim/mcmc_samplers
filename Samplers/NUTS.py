import tensorflow as tf

from Samplers.sampler import Sampler

class NUTS(Sampler):
    """
    An implementation of the No-U-Turn Sampler. This MCMC sampler is a modified
    HMC sampler where for any given step the differential equation solver
    is run until it starts to double back on itself. In order to ensure
    detailed balance is preserved, the expansions process is that of a binary
    tree, where the tree of is double in an arbitrary direction in each step
    until the path starts to double back anywhere along the tree.
    
    This implementation does not support multiple chains at the moment.
    """
    
    def __init__(self, dimensions, log_likelihood_fn, epsilon,  l, maxDepth=10,
                 parallelSamples = 1, dtype=tf.float32, mu = 100, gamma = 0.4,
                 t0 = 10, kappa = 0.75, target=0.6, dMax=1000):
        """
        The constructor for the NUTS sampler

        Parameters
        ----------
        dimensions : the number of dimensions of the sampled distribution
        log_likelihood_fn : a function accepting a state from the distribution
            and returning the natural logarithm of the probability of that
            state. This probability need not be normalized.
        epsilon : a parameter determining the step size for the differential
            equation solver
        l : number of leapfrog steps to perform for each sample
        maxDepth: maximum depth of tree until the algorithm terminates
        parallelSamples : Number of parallel chains to run
        dtype : data type of data, tf.float32 is recomended
        mu : Scaling factor on initial epsilon for dual-averaging
        gamma : Controls shrinkage towards mu for dual-averaging
        t0 : Greater than 0, stabalizes initial iterations  for dual-averaging
        kappa : controls step size schedule for dual-averaging
        target : target acceptance probability for dual-averaging 
        dMax : maximum energy violation

        Returns
        -------
        None.

        """
        self.dtype=dtype
        self.dimensions=dimensions
        self.log_likelihood_fn = log_likelihood_fn
        self.sampledVals = []
        self.acceptance=tf.cast(0, self.dtype)  
        self.epsilon = tf.cast(tf.ones((1,parallelSamples))*epsilon, self.dtype)
        self.logEpsilonBar=tf.cast(tf.zeros((1,parallelSamples)), self.dtype)
        self.mu = tf.cast(tf.math.log(mu*self.epsilon), self.dtype)
        self.gamma=tf.cast(gamma, self.dtype)
        self.t0 = tf.cast(t0, self.dtype)
        self.kappa=tf.cast(kappa, self.dtype)
        self.h = tf.cast(0*self.epsilon, self.dtype)
        self.target=tf.cast(target, self.dtype)
        self.L = l
        self.maxDepth = maxDepth
        self.dMax = dMax
        
    

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
        acceptance = tf.cast(0, self.dtype)
        
        #Use float32 for ease of use with GPUs
        currentState = tf.cast(initial_state, self.dtype) 
        
        #run sampler for the number of burn_in steps
        i = tf.constant(0)
        samples = self.burn_in
        condition = lambda i, currentState, h, logEpsilonBar, epsilon: tf.less(i, samples)
        
        #Body of while loop for burn_in, no states or acceptance info kept
        def body(i, currentState, h, logEpsilonBar, epsilon):
            m=tf.cast(i+1, self.dtype) #Step number
            
            #Progress one step
            currentState, accept, epsilon = self.one_step(currentState, epsilon)
           
            #Update h val
            h = (1-1/(m+self.t0))*h+(1/(m+self.t0))*(self.target-accept)
            
            #Update log Epsilon
            logEpsilon = self.mu-h*(m**0.5)/self.gamma
            
            #Update log Epsilon bar
            logEpsilonBar = m**(-self.kappa)*logEpsilon+(1-m**(-self.kappa))*logEpsilonBar
            
            #Set epsilon
            epsilon = tf.math.exp(logEpsilon)
            
            return([tf.add(i, 1), currentState, h, logEpsilonBar, epsilon])
        
        
        h, logEpsilonBar, epsilon = self.h, self.logEpsilonBar, self.epsilon
        
        #tf.while_loop to speed up sampling
        i, currentState,h, logEpsilonBar, epsilon = tf.while_loop(condition, body, 
                                                     [i, currentState,
                                                      h, logEpsilonBar, epsilon])
        
        epsilon=tf.exp(logEpsilonBar)
        tf.print("Final step size", epsilon)
        
        #tensorArray of set size to store samples
        sampledVals = tf.TensorArray(self.dtype, size=self.samples,
                                     dynamic_size=False)
        
        
        
        #run sampler for the number of sampling steps
        samples += self.samples
        def condition(i, currentState, sampledVals, acceptance, h, logEpsilonBar, epsilon):
            return(tf.less(i, samples))
        
        #Body of while loop for sampling, states and acceptance info kept
        def body(i, currentState, sampledVals, acceptance, h, logEpsilonBar, epsilon):
            
            currentState, accept, epsilon = self.one_step(currentState, epsilon)
            
            acceptance+=tf.reduce_sum(accept)
            sampledVals= sampledVals.write(i-self.burn_in, currentState)
            
            
            return([tf.add(i, 1), currentState, sampledVals,
                    acceptance, h, logEpsilonBar, epsilon])

        #tf.while_loop to speed up sampling
        i, currentState, sampledVals,acceptance, h, logEpsilonBar, epsilon = \
            tf.while_loop(condition, body, [i, currentState, sampledVals, acceptance, h, logEpsilonBar, epsilon])
        
        #trun sampledVals into a normal Tensor
        sampledVals = sampledVals.stack()                                                                                    
        
        return(sampledVals, acceptance)

     
    def energy(self, position, momentum):
        """
        Get the total energy of a location and momentum
        """
        potential = -self.log_likelihood_fn(position)
        kinetic = tf.reduce_sum(tf.square(momentum)/2, axis=0)
        total = potential + kinetic
        return(total)
    
    def getUGradient(self, position):
        """
        Gradient at a location
        """
        energy = None
        with tf.GradientTape() as g:
          g.watch(position)
          energy = -self.log_likelihood_fn(position)
        grad = g.gradient(energy, position)
        return(grad, energy)

    def one_step(self, currentState, epsilon):
        """
        A function which performs one step of the NUTS sampler

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        epsilon : the current step size

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedProb : acceptance probability for each chain
        accepted : 1 if the new state was accepted, 0 otherwise for each chain
        
        """
        
        newState, epsilon, accepted= self.propose_states(currentState, epsilon)
        updatedState, _ = self.accept_reject(newState)
        
        
        return(updatedState, accepted,  epsilon)

        
        
    def propose_states(self, currentState, epsilon):
        """
        

        Parameters
        ----------
        currentState : the current state of the sampler for each chain
        epsilon : current step size

        Returns
        -------
        state : the proposed new state of the sampler for each chain
        epsilon : current step size
        acceptance : acceptance rate of states in chain

        """
        
        
        def nutsCond(depth, state, momentum, energy, stateLeft, momentumLeft,
                     stateRight, momentumRight, sliceVal, stop, currentCount,
                     currentRatio, epsilon, fullCount, startE):
            
            diff = stateRight-stateLeft
            
            #Check for doubling back
            stop = tf.where(tf.reduce_sum(diff*momentumRight, axis=0)<=0 or 
                            tf.reduce_sum(diff*momentumLeft, axis=0)<=0,
                            0*stop,stop)
            
            #Check for significant deviation from slice
            stop = tf.where(sliceVal<tf.exp(self.dMax-energy),stop,0*stop)
            
            #Don't stop at zero depth
            stop = tf.where(depth==depth*0, stop*0+1, stop)
            
            #Check that a stopping condition hasn't been met
            return(depth<self.maxDepth and tf.equal(stop, stop*0+1))

        def nutsBody(depth, state, momentum, energy, stateLeft, momentumLeft,
                     stateRight, momentumRight, sliceVal, stop, currentCount,
                     currentRatio, epsilon, fullCount, startE):
            """
            Main No U-turn method

            """
            
            #Starting values
            initialState = state
            initialMomentum = momentum
            initialEnergy = energy
        
        
        
        
            def one_leapfrog_step(i, currentState, currentMomentum, 
                                  currentEnergy, states, momenta, energies,
                                  direction, count, sliceVal, epsilon):
                """
                Run one leapfrog step
                """
                grad,_ = self.getUGradient(currentState)
                halfMomentum = currentMomentum - direction*(epsilon/2)*grad
                newState = currentState + direction*epsilon*halfMomentum
                
                grad, _ = self.getUGradient(newState)
                newMomentum = halfMomentum - (direction*epsilon/2)*grad
                
                energy = self.energy(newState, newMomentum)
                
                #Store states and momentum and enrgy check whether state
                #is in slice
                states=states.write(i, newState)
                momenta=momenta.write(i, newMomentum) 
                countShape=count.shape
                count = tf.where(sliceVal<tf.exp(-energy),count+1,count)
                count=tf.reshape(count, countShape)
                
                energies = energies.write(i, energy)
                return(tf.add(i,1),newState, newMomentum, energy, states,
                       momenta, energies, direction, count, sliceVal, epsilon)
            
            def condition(i, currentState, currentMomentum, currentProb,
                          states, momenta, energies, directio, count,
                          sliceVal, epsilon):
                #Check whether the whole subtree has been filled out
                return(tf.less(i, 2**depth))
            
            
            #Storage for states
            states=tf.TensorArray(self.dtype, size=2**depth,
                                  dynamic_size=False, clear_after_read=False)
            momenta=tf.TensorArray(self.dtype, size=2**depth,
                                   dynamic_size=False, clear_after_read=False)
            energies=tf.TensorArray(self.dtype, size=2**depth,
                                    dynamic_size=False, clear_after_read=False)
            
            #Random starting directions and momentum, starting states
            randomDirection = tf.random.uniform(initialEnergy.shape,-1,1, 
                                                dtype = self.dtype)
            randomDirection = tf.cast(tf.where(randomDirection<0, -1, 1), 
                                      self.dtype)
            startState = tf.where(randomDirection<0, stateLeft, stateRight)
            startMomentum = tf.where(randomDirection<0, momentumLeft, momentumRight)
            
            #startE intiail value doesn't matter 
            count = tf.add(tf.constant(0),0)
            i = tf.constant(0)
            
            #Perform the sampling
            i, state, momentum, currentE,states, momenta, energies, \
                direction, count, sliceVal, epsilon = \
                    tf.while_loop(condition, 
                                  one_leapfrog_step, 
                                  [i, startState, startMomentum, initialE,
                                   states, momenta, energies, randomDirection,
                                   count, sliceVal, epsilon])
            
            
            def compBody(i, j, states, momenta, energies, depth, stop,
                         sliceVal, randomDirection):
                """
                Check whether any of the subtrees from the current tree branch
                satisfy the stopping conditions. This will iterate over all the
                end points of the subtrees.
                """
                
                #Get state info
                stopShape = stop.shape
                j = tf.where(depth==0, 1, j)
                oldState = states.read(i)
                oldMomentum = momenta.read(i)
                energy = energies.read(i)
                
                newState = states.read(i+j-1)
                newMomentum = momenta.read(i+j-1)
    
                diff = newState-oldState
                diff=diff*randomDirection
                
                #Doubling back
                stop = tf.where((tf.reduce_sum(diff*oldMomentum, axis=0)<=0 or
                                 tf.reduce_sum(diff*newMomentum, axis=0)<=0) and
                                depth>0,stop*0,stop)
                
                #Too far off
                stop = tf.where(sliceVal<tf.exp(self.dMax-energy),stop,stop*0)
                
                
                stop = tf.where(depth==0, 1+stop*0,stop)
                
                #Find the next set of states
                j = tf.where(depth==0, 2, j)
                j = j*2
                
                j = tf.where(i%j==0, j, 2**depth)
                
                i =  tf.where(depth==0, 2**depth, i)
                
                i = tf.where(tf.less(2**depth, i+j-1) and depth>0, i+2, i)
                
                j = tf.where(tf.less(2**depth, i+j-1) and depth>0, 2, j)
                
                stop = tf.reshape(stop, stopShape)
                return(i,j, states, momenta, energies, depth, stop, sliceVal, randomDirection)
             
                
            def compCondition(i, j, states, momenta, energies, depth, stop,
                              sliceVal, randomDirection):
                # Determine whether to keep running the checks
                return(tf.less(i, 2**depth) and stop == stop*0+1)
            
            
            #Check whether anything in the new subtree reaches the stopping
            #condition
            i = tf.constant(0)
            j = tf.constant(2)
            
            _, _, states, momenta, energies, depth, stop, sliceVal, \
                randomDirection = tf.while_loop(compCondition,
                                                compBody,
                                                [i, j, states, momenta,
                                                 energies, depth, stop,
                                                 sliceVal, randomDirection])
            
            
            #Get all the states and some random locations
            states = tf.transpose(tf.transpose(states.stack())[0])
            momenta = tf.transpose(tf.transpose(momenta.stack())[0])
            energies = tf.transpose(tf.transpose(energies.stack())[0])
            randomLocs = tf.random.uniform(initialEnergy.shape, 0, 2**depth, dtype=tf.int32)
            
            #Get valid locations
            valid = tf.cast(tf.where(sliceVal<tf.exp(-energies),1,0), self.dtype)
            valid = tf.expand_dims(valid,0)
            valid = tf.math.log(valid)
            
            #Determine acceptance rate of proposed locations
            acceptRates = tf.where(1<tf.exp(startE - energies),
                                   tf.cast(1, self.dtype),
                                   tf.exp(startE - energies))
            acceptRates = tf.where(tf.math.is_nan(acceptRates),
                                   tf.cast(0, self.dtype), acceptRates)
            
            #Get overall acceptance rate
            acceptRates = tf.reduce_sum(acceptRates, axis=0)
            
            #Running sum of probs
            currentRatio+=acceptRates
            
            #Running count of samples
            fullCount+=2**depth
            
            #Get a new random state from the accepted proposed states
            newRandom  = tf.random.categorical(valid,1)
            newRandom = tf.cast(newRandom, depth.dtype)
            newRandom = tf.where(newRandom==2**depth, 0, newRandom)
            
            randomState = tf.gather_nd(states, newRandom, batch_dims=0)
            
            randomMomenta = tf.reshape(tf.gather_nd(states, randomLocs, 
                                                    batch_dims=0),
                                       initialMomentum.shape)
            randomEnergy = tf.gather_nd(energies, randomLocs, batch_dims=0)
            
            #Randomly select new or old state
            randomSelect = tf.random.uniform(initialEnergy.shape,0,1,
                                             dtype=self.dtype)
            randomState = tf.reshape(randomState, initialState.shape)
            currentCount=currentCount+count
            
            ratio = count/currentCount
            ratio=tf.cast(ratio, randomSelect.dtype)
            
            #Update the states and the tree edges
            newState = tf.where(randomSelect>ratio or stop==0,
                                initialState, randomState)
            newMomentum = tf.where(randomSelect>ratio or stop==0,
                                   initialMomentum, randomMomenta)
            newEnergy = tf.where(randomSelect>ratio or stop==0,
                                 initialEnergy, randomEnergy)
            stateRight = tf.where(stop==0 or randomDirection<0,
                                  stateRight, tf.reshape(states[-1,:],
                                                         initialState.shape))
            momentumRight = tf.where(stop==0 or randomDirection<0,
                                     momentumRight, tf.reshape(momenta[-1,:],
                                                               initialState.shape))
            
            
            stateLeft = tf.where(stop==0 or randomDirection>0,
                                 stateLeft, tf.reshape(states[-1,:], 
                                                       initialState.shape))
            
            momentumLeft = tf.where(stop==0 or randomDirection>0,
                                    momentumLeft,
                                    tf.reshape(momenta[-1,:],
                                               initialState.shape))
            
            
            depth = depth + 1 
            return(depth, newState, newMomentum, newEnergy, stateLeft,
                   momentumLeft, stateRight, momentumRight, sliceVal, stop,
                   currentCount, currentRatio, epsilon, fullCount, startE)
        
        #Staring values
        momentum = tf.random.normal(currentState.shape, dtype=self.dtype)
        initialE = self.energy(currentState, momentum)
        sliceVal = tf.random.uniform(initialE.shape, 0, tf.exp(-initialE), 
                                     dtype=self.dtype)
        stateLeft = currentState
        stateRight = currentState
        state=currentState
        momentumLeft = momentum
        momentumRight = momentum
        depth = tf.zeros((), dtype=tf.int32)
        stop = tf.ones(initialE.shape)
        currentCount = tf.constant(1)    
        fullCount = tf.constant(0)
        currentRatio = tf.constant(0, dtype=self.dtype)
        startE = initialE
        
        #Run the sampler
        depth, state, momentum, energy, stateLeft, momentumLeft,\
            stateRight, momentumRight, sliceVal, stop, currentCount, \
            currentRatio, epsilon, fullCount, \
            startE = tf.while_loop(nutsCond,
                                   nutsBody,
                                   [depth, state, momentum, initialE,
                                    stateLeft, momentumLeft, stateRight,
                                    momentumRight, sliceVal, stop,
                                    currentCount, currentRatio, epsilon,
                                    fullCount, startE])
            
        currentCount = tf.cast(fullCount, self.dtype)
        acceptance = currentRatio/currentCount
        return(state, epsilon, acceptance)
    
    def accept_reject(self, newState):
        """
        A state is always accepted

        """
        
        
        return(newState, tf.ones_like(newState))
    