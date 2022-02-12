# Optimising Optical Circuits for Purification Resource States

## 0. Background

In this notebook, we use machine learning to optimise the parameters of an optical circuit towards generating high quality quantum resource states (in particular, we employ the global optimisation algorithm *basinhopping*). These resource states are needed to perform entanglement purification in a linear-optical system, as discussed in our paper "Achieving the ultimate end-to-end rate of a lossy quantum communication network" [1]. We have separated this notebook into four sections: 

Section 1: All the necessary libraries are imported.  
Section 2: The functions that allows us to simulate an optical circuit are prepared, and then verified to check that they work.  
Section 3: The *basinhopping* algorithm is prepared, and an example optimisation is run to illustrate to the user how it works.  
Section 4: The best optimised parameter sets that we could find were saved; these parameter sets are loaded and detailed in this section.  

Finally, we acknowledge the papers "Production of photonic universal quantum gates enhanced by machine learning" [[2](https://doi.org/10.1103/PhysRevA.100.012326)] and "Progress towards practical qubit computation using approximate Gottesman-Kitaev-Preskill codes" [[3](https://doi.org/10.1103/PhysRevA.101.032315)], whose code we have used and modified here for our own purposes. We also acknowledge the libraries *strawberryfields* [[4](https://doi.org/10.22331/q-2019-03-11-129)] and *thewarlus* [[5](https://doi.org/10.21105/joss.01705)], which we used to perform the quantum simulation. 

[1] M. S. Winnel, J. J. Guanzon, N. Hosseinidehaj, and T. C. Ralph, "Achieving the ultimate end-to-end rate of a lossy quantum communication network," *to be published* (2021). \
[2] K. K. Sabapathy, H. Qi, J. Izaac, and C. Weedbrook, "Production of photonic universal quantum gates enhanced by machine learning," [Physical Review A **100**, 012326 (2019)](https://doi.org/10.1103/PhysRevA.100.012326). \
[3] I. Tzitrin, J. E. Bourassa, N. C. Menicucci, and K. K. Sabapathy, "Progress towards practical qubit computation using approximate Gottesman-Kitaev-Preskill codes," [Physical Review A **101**, 032315 (2020)](https://doi.org/10.1103/PhysRevA.101.032315). \
[4] N. Killoran, J. Izaac, N. Quesada, V. Bergholm, M. Amy, and C. Weedbrook, "Strawberry Fields: a software platform for photonic quantum computing," [Quantum **3**, 129 (2019)](https://doi.org/10.22331/q-2019-03-11-129). \
[5] B. Gupt, J. Izaac, and N. Quesada, "The Walrus: a library for the calculation of hafnians, Hermite polynomials and Gaussian boson sampling," [Journal of Open Source Software **4**, 1705 (2019)](https://doi.org/10.21105/joss.01705).  


## 1. Libraries


```python
### Date and time
from datetime import datetime  # For current day and time
from time import time  # For runtime of scripts

### Saving and viewing data with current working directory
import pickle
import os
cwd = os.getcwd()

### Math and numerics
import numpy as np
from numpy import pi
from scipy.optimize import basinhopping, minimize

### Quantum simulation packages
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Sgate
from thewalrus.quantum import state_vector, density_matrix_element

```

## 2. Circuit Simulation 



```python
def apply_interferometer(thetas, phis, nmodes, q):
    '''Applies beamsplitters in a rectangular array to the modes of a 
    circuit.
    
    Args:
        thetas (array): the beamsplitter angles
        phis (array): the beamsplitter phases
        nmodes (int): the number of circuit modes
        q (tuple): the circuit modes
    
    Returns:
        None
    '''
    # The i's are the rows; the j's are the columns; the k's are
    # the parameter indices. The maximum values the indices are taken
    # by observing the pattern in the rectangular decomposition.
    k = 0
    for i in range(nmodes):
        if i % 2 == 0:       
            for j in range(int(np.floor(nmodes/2))):
                BSgate(thetas[k], phis[k]) | (q[2*j], q[2*j + 1])
                k += 1
        else:
            for j in range(1, int(np.floor((nmodes-1)/2)) + 1):
                BSgate(thetas[k], phis[k]) | (q[2*j-1], q[2*j])
                k += 1

engine_time = 0  # Engine run time tracker. 
hafnian_time = 0  # Hafnian run time tracker. 

def fock_circuit(params, target, post_m, nmodes):
    '''Runs a constrained variational circuit with specified 
    parameters to generate a Fock superposition in the final mode.
    Assumes the circuit squeezes, displaces and then sends them 
    through an interferometer in a rectangular arrangement. We assume no
    phases in the squeezing and displacement.

    Args:
        params (list): list of gate parameters for the constrained
            variational quantum circuit, with nmodes modes.
            This should contain the following values: 
            * sq_r1, sq_r2, ..., sq_rn: the squeezing magnitudes applied 
                to all the modes
            * ds_r1, ds_r2, ..., ds_rn: the displacement magnitudes applied 
                to all the modes
            * bs_theta1, bs_theta2, ..., bs_theta[n*(n-1)]/2: the 
                beamsplitter angles theta
            * bs_phi1, bs_phi2, ..., bs_theta[n*(n-1)]/2: the 
                beamsplitter phases phi
        target (array): the Fock coefficients of the target state
        post_m (list): the Fock state measurement outcomes to be 
            post-selected; post_m[i] corresponds to the ith mode
        nmodes (int): the number of circuit modes

    Returns:
        tuple: the output fidelity to the target state, the probability 
        of post-selection, and the output state in Fock basis. 
    '''
    nsplitters = nmodes * (nmodes - 1) // 2  # Number of beam splitters
    ndetmodes = len(post_m)  # Number of modes with detections
    target_dim = len(target)  # Dimension of the target state.
                                                                                                                                            
    # Unpack the circuit parameters
    sq_r = params[:nmodes]  # Squeezing r's
    ds_r = params[nmodes:2*nmodes]  # Displacement r's 
    bs_thetas = params[2*nmodes:2*nmodes + nsplitters]  # BS theta's
    bs_phis = params[2*nmodes + nsplitters:]  # BS phi's

    # Prepare the program
    prog = sf.Program(nmodes)
    with prog.context as q:
        for k in range(nmodes):
            Sgate(sq_r[k], 0) | q[k]
            Dgate(ds_r[k], 0) | q[k]
        apply_interferometer(bs_thetas, bs_phis, nmodes, q)
        
    # Run program via engine, extract state, means, and covariances.
    start = time()
    eng = sf.Engine('gaussian')
    state = eng.run(prog).state
    mu, cov = state.means(), state.cov()
    global engine_time
    engine_time += time() - start
    
    # Implements post-selection of measurement outcomes
    start = time()
    mu_r, cov_r = state.reduced_gaussian(list(range(ndetmodes)))
    prob = density_matrix_element(mu_r, cov_r, post_m, post_m).real
    postdict = {i: post_m[i] for i in range(ndetmodes)} 
    rho = state_vector(mu, cov, post_select=postdict, cutoff=target_dim)
    global hafnian_time
    hafnian_time += time() - start

    # Normalize the output state and calculates fidelity
    rho = rho / np.sqrt(prob)
    fid = np.abs(np.sum(np.conj(target)*rho)) ** 2
    
    if prob < 1e-15:
        return 0, 0, rho
    return fid, prob, rho

def to_r(dB):
    '''Returns the squeezing magnitude in dB as operator parameter r.'''
    return np.log(10) * dB / 20

def to_db(r):
    '''Returns the value of squeezing parameter r in dBs.'''
    return -20 * r / np.log(10)

def init(clip_size):
    '''Generates an initial random parameter.

    Args:
        clip_size (float): the parameter will be clipped
            to the domain [-clip_size, clip_size].

    Returns:
        float: the random clipped parameter.
    '''
    return np.random.rand() * 2 * clip_size - clip_size

def random_params(rmax_dB, dsmax, nmodes):
    ''' Returns a list of random parameters for an nmodes-mode circuit.
    
    Args:
        rmax_dB (float): the maximum squeezing in dBs
        dsmax (float): the maximum displacement
        nmodes: the number of modes in the circuit
        
    Returns:
        list: nmodes squeezing parameters followed by
              nmodes displacement parameters followed by
              nmodes*(nmodes-1) angles for beamsplitters.
    '''
    # The squeezing bound.
    clip = abs(to_r(rmax_dB))
    # Put the squeezing, displacement and angle bounds into a list.
    bound = [clip]*nmodes + [dsmax]*nmodes + [pi]*nmodes*(nmodes - 1)
    # Randomize
    return list(map(init, bound))

### Verification tests for above functions
if __name__ == '__main__':
    print(50*':' + '\nVerification Test of Circuit Simulator Functions',
          '\nStart time: ', datetime.now())
    
    print('\nChosen user input...')
    cutoff = 4  # Cut-off Fock state
    ntarmodes = 2  # Number of modes of target state
    target = np.zeros([cutoff]*ntarmodes)
    target[1][0] = 1 
    target[0][1] = 1 
    target = target/np.linalg.norm(target)  # Normalize
    post_m = [1,1,1]  # Post-selection measurement pattern
    nmodes = len(post_m) + ntarmodes
    print('Target state:\n'+np.array_str(target, precision=2),
          '\nPost-selection pattern: {} \nCircuit modes: {}'.format(post_m,nmodes))
    
    print('\nVerifying random parameter set generator...')   
    params = random_params(12, 1, nmodes)
    print('Random parameters:\n'+np.array_str(np.array(params), precision=3))

    print('\nVerifying circuit simulator and fidelity...') 
    start = time() 
    fid, prob, rho = fock_circuit(params, target, post_m, nmodes)
    time_elapsed = time() - start 
    print('Output state:\n'+np.array_str(rho, precision=2),
          '\nFidelity: {:.5f} \nProbability: {:.5f}'.format(fid, prob))
    
    print('\nTotal Walrus time: {:.3f} seconds'.format(hafnian_time),
          '\nTotal StrawberryFields time: {:.3f} seconds'.format(engine_time),
          '\nTotal run time: {:.3f} seconds'.format(time_elapsed))
    
```

    ::::::::::::::::::::::::::::::::::::::::::::::::::
    Verification Test of Circuit Simulator Functions 
    Start time:  2022-02-12 17:17:49.102659
    
    Chosen user input...
    Target state:
    [[0.   0.71 0.   0.  ]
     [0.71 0.   0.   0.  ]
     [0.   0.   0.   0.  ]
     [0.   0.   0.   0.  ]] 
    Post-selection pattern: [1, 1, 1] 
    Circuit modes: 5
    
    Verifying random parameter set generator...
    Random parameters:
    [-1.120e+00  1.954e-01  2.077e-01  4.136e-01  5.471e-01  1.253e-01
      2.506e-03 -2.312e-01 -6.896e-02 -1.593e-01  2.023e+00  2.001e+00
     -1.094e+00  3.040e+00 -3.010e+00  3.988e-01  1.022e+00  6.551e-02
     -2.574e+00 -4.133e-01 -1.035e+00 -1.593e+00 -1.367e+00  1.697e-01
      2.224e+00  5.862e-02 -2.114e+00 -1.182e+00 -2.141e+00 -2.202e+00]
    
    Verifying circuit simulator and fidelity...
    Output state:
    [[ 3.07e-01-0.17j  6.35e-02-0.1j  -9.63e-02+0.03j -3.62e-02+0.06j]
     [ 5.94e-01+0.41j  1.31e-01+0.12j -1.85e-01-0.12j -7.61e-02-0.07j]
     [ 2.90e-02+0.01j -1.77e-03+0.04j -1.43e-02+0.01j -5.85e-04-0.02j]
     [-3.45e-01+0.17j -7.85e-02+0.02j  1.04e-01-0.06j  4.45e-02-0.01j]] 
    Fidelity: 0.26269 
    Probability: 0.01362
    
    Total Walrus time: 0.026 seconds 
    Total StrawberryFields time: 0.007 seconds 
    Total run time: 0.035 seconds
    

## 3. Circuit Optimisation



```python
def cost(fid, prob, costn): 
    '''Different cost functions.'''
    if costn == 1:
        return -fid -0.1*prob
    elif costn == 2:
        return -fid -prob
    elif costn == 3:
        return -0.1*fid -prob
    elif costn == 4:
        return -prob
    else: 
        return -fid
    
def cost_cal(params, target, post_m, nmodes, costn):
    '''Returns the cost of the constrained variational circuit.
    Minimising the cost function will result in the output 
    state approaching the target state with good probability.

    Args:
        params (list): list of gate parameters for the constrained
            variational quantum circuit, with nmodes modes.
        target (array): the Fock coefficients of the target state. 
        post_m (list): the Fock state measurement outcomes to be 
            post-selected; post_m[i] correpsonds to the ith mode. 
        nmodes (int): the number of circuit modes. 
        costn (int): the chosen cost function. 

    Returns:
        float: cost value.
    '''
    fid, prob, _ = fock_circuit(params, target, post_m, nmodes)
    return cost(fid, prob, costn)

engine_time = 0  # Engine run time tracker. 
hafnian_time = 0  # Hafnian run time tracker.
fid_i = 0  # Fidelity step index for optimization.
cost_best = 0  # Best cost found by optimization.

def run_global_optimization(target, post_m, nmodes, costn,
                            init_guess=None,
                            niter=50,
                            max_iter=1000,
                            direc='test_data\\',
                            save=False):
    '''Run the constrained variational quantum circuit global 
    optimization using the basin hopping algorithm.

    Args:
        target (array): the Fock coefficients of the target state.
        post_m (list): the Fock state measurement outcomes to be 
            post-selected; post_m[i] correpsonds to the ith mode.
        nmodes (int): the number of modes of the circuit. 
        costn (int): the chosen cost function. 
        init_guess (array): supplies the initial guess to the 
            optimization; if None, randomizes the guess.
        niter (int): the number of hops in basinhopping.
        max_iter (int): the maximum number of iterations for minimize.
        direc (str): data directory to save output.
        save (bool): if True, save the output in directory direc. 

    Returns:
        tuple: optimization results. A tuple of circuit parameters,
            fidelity to the target state, and probability of generating 
            the state.
    '''
    # Generate the initial random parameters and bounds.
    clip = 15 # Maximum squeezing in dB to search through.
    disp = 1 # Maximum displacement to search through.
    if init_guess == None:
        init_guess = random_params(clip, disp, nmodes)
    # Search limits for squeezing, displacement and BS parameters.
    lims = [(-to_r(clip), to_r(clip))]*nmodes + [(-disp, disp)]*nmodes + [(-pi, pi)]*nmodes*(nmodes - 1)
    # Perform the optimization
    minimizer_kwargs = {'method': 'SLSQP',  # or L-BFGS-B
                        'args': (target, post_m, nmodes, costn),
                        'options': {'maxiter': max_iter},
                        'bounds': lims}
    
    # Output file name
    m_str = '&'.join([str(outcome) for outcome in post_m])
    res_str = ('paratest_(nmodes={})(pattern={})'.format(nmodes, m_str) + '.pickle')
    out_file_name = (direc + res_str) 
    def myAccept(xk, f, accepted):
        '''This accept condition will save the current best parameters.'''
        global fid_i
        global cost_best
        fid_i += 1
        fid, prob, rho = fock_circuit(xk, target, post_m, nmodes)
        cost_current = cost(fid, prob, costn)
        print('Hop {} fidelity: {:.5f}, probability: {:.3e}, cost: {:.5f}'
              ''.format(fid_i, fid, prob, cost_current))
        if save and cost_current < cost_best:
            with open(out_file_name, 'wb') as handle:
                pickle.dump([xk, [fid, prob]], handle, protocol=-1)
            print('This is the best so far. Saving as: ' + res_str)
            cost_best = cost_current

    print('Optimizing a {}-mode circuit for target state'.format(nmodes),
          '\nwith measurement pattern {} of the first {} modes.\nBasin hopping '
          'called with {} iterations and max of {} iterations for minimize.'
          ''.format(post_m, len(post_m), niter, max_iter))
    res = basinhopping(cost_cal, init_guess, minimizer_kwargs=minimizer_kwargs,
                       niter=niter, callback=myAccept)

    fid, prob, rho = fock_circuit(res.x, target, post_m, nmodes)
    print('Optimised parameters:\n'+np.array_str(np.array(res.x), precision=2),
          '\nOutput state (abs):\n'+np.array_str(np.abs(rho), precision=2, suppress_small=True),
          '\nFidelity: {:.5f} \nProbability: {:.5f}'.format(fid, prob))

    return res.x, fid, prob

### Verification tests for above functions
if __name__ == '__main__':
    print(50*':' + '\nVerification Test of Basin Hopping Global Search for High Fidelity',
          '\nStart time: ', datetime.now()) 

    print('\nChosen user input...')
    cutoff = 2  # Cut-off Fock state
    ntarmodes = 2  # Number of modes of target state
    target = np.zeros([cutoff]*ntarmodes)

    target[0][1] = 1 
    target[1][0] = 1 
    
    #target[0][1][1] = 1 
    #target[1][0][0] = 1 
    
    #target[0][2][0] = 1 
    #target[1][1][1] = 1/2 
    #target[2][0][2] = 1 

    #target[0][1][1][1] = 1 
    #target[1][0][0][0] = 1 
    
    target = target/np.linalg.norm(target)  # Normalize
    post_m = [1]  # Post-selection measurement pattern
    nmodes = len(post_m) + ntarmodes  # Number of circuit modes 
    print('Target state:\n'+np.array_str(target, precision=2),
          '\nPost-selection pattern: {} \nCircuit modes: {}'.format(post_m,nmodes))

    print('\nVerifying circuit global optimisation...') 
    start = time() 
    params, fid, prob = run_global_optimization(target, post_m, nmodes, costn=1, niter=5, max_iter=1000, save=True, init_guess=None, direc=cwd+"\\statefinderdata\\")
    time_elapsed = time() - start 
    
    print('\nTotal Walrus time: {:.3f} seconds'.format(hafnian_time),
          '\nTotal StrawberryFields time: {:.3f} seconds'.format(engine_time),
          '\nTotal run time: {:.3f} seconds'.format(time_elapsed))

```

    ::::::::::::::::::::::::::::::::::::::::::::::::::
    Verification Test of Basin Hopping Global Search for High Fidelity 
    Start time:  2022-02-12 17:17:49.198831
    
    Chosen user input...
    Target state:
    [[0.   0.71]
     [0.71 0.  ]] 
    Post-selection pattern: [1] 
    Circuit modes: 3
    
    Verifying circuit global optimisation...
    Optimizing a 3-mode circuit for target state 
    with measurement pattern [1] of the first 1 modes.
    Basin hopping called with 5 iterations and max of 1000 iterations for minimize.
    Hop 1 fidelity: 0.99994, probability: 5.606e-04, cost: -0.99999
    This is the best so far. Saving as: paratest_(nmodes=3)(pattern=1).pickle
    Hop 2 fidelity: 1.00000, probability: 2.500e-01, cost: -1.02500
    This is the best so far. Saving as: paratest_(nmodes=3)(pattern=1).pickle
    Hop 3 fidelity: 1.00000, probability: 2.500e-01, cost: -1.02500
    Hop 4 fidelity: 1.00000, probability: 2.500e-01, cost: -1.02500
    This is the best so far. Saving as: paratest_(nmodes=3)(pattern=1).pickle
    Hop 5 fidelity: 1.00000, probability: 2.500e-01, cost: -1.02500
    Optimised parameters:
    [-8.82e-01  8.82e-01 -2.80e-05  5.16e-05  6.72e-05 -6.81e-05 -7.85e-01
      7.85e-01  5.19e-05 -3.14e+00 -1.58e-04 -6.11e-01] 
    Output state (abs):
    [[0.   0.71]
     [0.71 0.  ]] 
    Fidelity: 1.00000 
    Probability: 0.25000
    
    Total Walrus time: 7.518 seconds 
    Total StrawberryFields time: 3.360 seconds 
    Total run time: 11.609 seconds
    

## 4. Load Previously Optimised Parameters



```python
if __name__ == '__main__': 
    print(50*':' + '\nLoad Previously Found Parameter Sets') 
    
    # Chose saved files 
    loadn = 1
    if loadn == 1:  
        # Best for |011> + |100> with fidelity 0.99983 and probability 2.11e-06
        file = open(cwd+"\\statefinderdata\\"+"para_011_100.pickle",'rb')
        cutoff = 2  # Cut-off Fock state
        ntarmodes = 3  # Number of modes of target state
        target = np.zeros([cutoff]*ntarmodes)
        target[0][1][1] = 1 
        target[1][0][0] = 1 
        post_m = [1,1,1]  # Post-selection measurement pattern
    elif loadn == 2:  
        # Best for |020> + |111>/2 + |202> with fidelity 0.91887 and probability 5.60e-10
        file = open(cwd+"\\statefinderdata\\"+"para2_020_111_202.pickle",'rb')
        cutoff = 3  # Cut-off Fock state
        ntarmodes = 3  # Number of modes of target state
        target = np.zeros([cutoff]*ntarmodes)
        target[0][2][0] = 1 
        target[1][1][1] = 1/2
        target[2][0][2] = 1      
        post_m = [1,1,1,1,1]  # Post-selection measurement pattern
    elif loadn == 3:  
        # Best for |0111> + |1000> with fidelity 0.98332 and probability 9.86e-08
        file = open(cwd+"\\statefinderdata\\"+"para3_0111_1000.pickle",'rb')
        cutoff = 2  # Cut-off Fock state
        ntarmodes = 4  # Number of modes of target state
        target = np.zeros([cutoff]*ntarmodes)
        target[0][1][1][1] = 1 
        target[1][0][0][0] = 1
        post_m = [1,1,1,1,1]  # Post-selection measurement pattern
    
    # Load saved files
    print('\nLoading best parameter set #{}...'.format(loadn))
    loadfile = pickle.load(file)
    file.close()
    params = loadfile[0]
    target = target/np.linalg.norm(target)  # Normalize
    print('Target state:\n'+np.array_str(target, precision=2),
          '\nPost-selection pattern: {} \nCircuit modes: {}'.format(post_m,nmodes))
    nmodes = len(post_m) + ntarmodes  # Number of circuit modes 

    # Unpack the circuit parameters
    nsplitters = nmodes * (nmodes - 1) // 2  # Number of beam splitters                                                                                                                            
    sq_r = params[:nmodes]  # Squeezing r's
    ds_r = params[nmodes:2*nmodes]  # Displacement r's 
    bs_thetas = params[2*nmodes:2*nmodes + nsplitters]  # BS theta's
    bs_phis = params[2*nmodes + nsplitters:]  # BS phi's
    print("Squeezing r in db:\n",np.array_str(np.array(to_db(sq_r)), precision=2),
          "\nDisplacement alpha:\n",np.array_str(np.array(ds_r), precision=2),
          "\nBeamsplitter theta:\n",np.array_str(np.array(bs_thetas), precision=2),
          "\nBeamsplitter phi:\n",np.array_str(np.array(bs_phis), precision=2))
    
    # Save as other files 
    #np.savetxt(cwd+"\\statefinderdata\\"+"para_r.csv", sq_r, delimiter=',')
    #np.savetxt(cwd+"\\statefinderdata\\"+"para_alpha.csv", ds_r, delimiter=',')
    #np.savetxt(cwd+"\\statefinderdata\\"+"para_theta.csv", bs_thetas, delimiter=',')
    #np.savetxt(cwd+"\\statefinderdata\\"+"para_phi.csv", bs_phis, delimiter=',')
    
    # Verify loaded circuit parameter
    print('\nVerifying loaded parameter set...') 
    fid, prob, rho = fock_circuit(params, target, post_m, nmodes)
    print('Output state (abs):\n'+np.array_str(np.abs(rho), precision=2, suppress_small=True),
          '\nFidelity: {:.5f} \nProbability: {:.2e}'.format(fid, prob))
    
```

    ::::::::::::::::::::::::::::::::::::::::::::::::::
    Load Previously Found Parameter Sets
    
    Loading best parameter set #1...
    Target state:
    [[[0.   0.  ]
      [0.   0.71]]
    
     [[0.71 0.  ]
      [0.   0.  ]]] 
    Post-selection pattern: [1, 1, 1] 
    Circuit modes: 3
    Squeezing r in db:
     [-6.08  0.09 -8.52 -1.65 -0.01  0.01] 
    Displacement alpha:
     [-0.34 -0.01 -0.19  0.05 -0.   -0.  ] 
    Beamsplitter theta:
     [ 0.15 -0.13 -2.2  -1.63  3.11  1.9   1.64 -1.65 -0.87 -2.73 -1.99 -0.24
     -1.59  2.55 -1.52] 
    Beamsplitter phi:
     [ 2.68 -2.34  0.58 -2.82  1.67  1.25 -0.44  1.86  2.24  1.06  1.18 -0.75
      2.3   0.6  -2.75]
    
    Verifying loaded parameter set...
    Output state (abs):
    [[[0.   0.  ]
      [0.   0.71]]
    
     [[0.71 0.  ]
      [0.   0.  ]]] 
    Fidelity: 0.99983 
    Probability: 2.11e-06
    
