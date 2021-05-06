# mcmc_samplers
A collection of implementations of various Markov Chain Monte Carlo samplers

In this repository are implementations of the following samplers:
- Metropolis-Hastings
- Gibbs
- Mirror Slice Sampler
- Multi Try Metropolis
- Langevin
- Hamiltonian Monte Carlo
- No-U-Turn Sampler
- Riemann Manifold Hamiltonian Monte Carlo
- l2hmc

All of the samplers are located in the Sampler folder. The sampler.py file contains a parent Sampler class with the basic details of how the samplers are implemented, and each sampler is contained in its own aptly named file. The only excpetion is HMC, as there are two clases in it, one called HMCLeapfrog using the second order leapfrog differential equation solver, and one called HMCPERFL, using the fourth order PERFL integrator. In the header of each of the python files is a description of the algorithm and the doctstring for the constructors explains the input parameters. 

For an example of how to use these samplers, see testSampler.py and testGibbs.py. The testSampler.py code shows how the Metropolis-Hastings sampler is used, but all the other samplers are used the same way. The only slight differences is that some require additional arguments. The exception is the Gibbs sampler, as it requires a function to supply samples from conditional distributions instead of evaluate the probability of states. This method is shown in testGibbs.py.

Many of the samplers use the Dual-Averaging method to set their step sizes. This means that for these samplers you need only supply a reasonable step size and an appropriate target acceptance to obtain good step sizes. The samplers which do not implement this either do not have a step size, such as the Gibbs sampler, use a different method, as in the l2hmc sampler, or are ill suited to it as the acceptance probabilities are 0 or 1 for the Mirror Slice Sampler.

Additionally, all of these samplers can handle multiple parallel sampling chains except for the No-U-Turn Sampler and the Riemann Manifold HMC sampler. It is recomended to use many parallel samples, as this can significantly boost performance with minimal additional computational cost. Additionally, as these samplers were implemented in TensorFlow, they can easily be ran on GPUs. Especially in the case of many independent chains, these samplers are very well suited for use on GPUs.

In order to actually run the sampler TensorFlow, numpy, and matplotlib must be installed. This can be accomplished through the command

```
pip install tensorflow numpy matplotlib
```