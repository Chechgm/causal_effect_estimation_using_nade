#! ./data/fake_data.py
""" This script creates a fake dataset that resembles that of the kideny stone example.

TODO: check the available functions.
TODO: change print for a logger.

The available functions are:
- save_data
- ks_binary_simulator
- binary_data_check
- ks_cont_recovery_simulator
-...
"""


import argparse
import numpy  as np
import os.path


def save_data(data, path, file_name):
    """Saves the data in the desired path, if it doesn't exist.
    """
    if not os.path.exists(os.path.join(path, file_name)):
        np.save(os.path.join(path, file_name), data)
        print(f"{file_name} saved succesfully")
    else:
        print(f"{file_name} already existed")
        

###############################################################################
###          Binary data simulator and Simpson's paradox "Checker"          ###
###############################################################################
def ks_binary_simulator(n=5000):
    """Creates data that resembles the kidney stone data set.
    
    Args:
        n: Number of observations to be generated
    Return:
        A numpy array with three columns. Size of kidney stone (if 1 Large)
          Treatment assigned (if 1 A), Recovery status (if 1 recovered)
    """
    p_l = 343/(343+357) # Probability of large kidney stones
    p_a_l = 263/343     # Probability of getting treatment a given small stones
    p_b_l = 80/343      # Probability of getting treatment b given small stones
    p_r_l_a = 192/263   # Probability of recovery given large stones and treatment a
    p_r_l_b = 55/80     # Probability of recovery given large stones and treatement b
    p_r_s_a = 81/87     # Probability of recovery given small stones and treatment a
    p_r_s_b = 234/270   # Probability of recovery given small stones and treatement b

    # Simulation
    l = np.random.binomial(1, p_l, size=(n,1))           # Simulation of kidney stone size
    a = np.random.binomial(1, l*p_a_l + (1-l)*(1-p_a_l)) # Simulation of treatment
    r = np.random.binomial(1, l*a*p_r_l_a + l*(1-a)*p_r_l_b + (1-l)*a*p_r_s_a + (1-l)*(1-a)*p_r_s_b)

    # Getting them together:
    data = np.hstack((l, a, r))

    return data


### Simpson's paradox checker
def binary_data_check(data):
    """
    This function checks whether the binary data generated follows Simpson's paradox.
    """
    # Check that P(R=1 | B) > P(R=1 | A)
    p_r_b = np.mean(data[data[:,1]==False, 2]) # P(R=1 | B)
    p_r_a = np.mean(data[data[:,1]==True, 2])  # P(R=1 | A)
    print("is P(R=1 | B) > P(R=1 | A)?", p_r_b > p_r_a)

    # Check that, individually, P(R=1 | A, S) > P(R=1 | B, S)
    #   and P(R=1 | A, L) > P(R=1 | B, L)

    mask_l_a = ((data[:,1]==True) & (data[:,0]==True))
    p_r_l_a  = np.mean(data[mask_l_a, 2])               # P(R=1 | A, L)
    mask_l_b = ((data[:,1]==False) & (data[:,0]==True))
    p_r_l_b  = np.mean(data[mask_l_b, 2])               # P(R=1 | B, L)
    print("is P(R=1 | A, L) > P(R=1 | B, L)? ", p_r_l_a > p_r_l_b)

    mask_s_a = ((data[:,1]==True) & (data[:,0]==False))
    p_r_s_a  = np.mean(data[mask_s_a, 2])               # P(R=1 | A, S)
    mask_s_b = ((data[:,1]==False) & (data[:,0]==False))
    p_r_s_b  = np.mean(data[mask_s_b, 2])               # P(R=1 | B, S)
    print("is P(R=1 | A, S) > P(R=1 | B, S)? ", p_r_s_a > p_r_s_b)


###############################################################################
###                      Continuous recovery simulator                      ###
###############################################################################
def ks_cont_recovery_simulator(n):
    """Creates data that resembles the kidney stone data set.
    
    Args:
        n: Number of observations to be generated
    Output:
        A numpy array with three columns. Size of kidney stone (if 1 Large)
          Treatment assigned (if 1 A), Recovery status (Normally distributed, depending on KS and T)
    """
    p_l = 343/(343+357) # Probability of large kidney stones
    p_a_l = 263/343     # Probability of getting treatment a given small stones
    p_b_l = 80/343      # Probability of getting treatment b given small stones
    TE    = 4           # Treatment effect

    # Simulation
    l = np.random.binomial(1, p_l, size=(n,1))           # Simulation of kidney stone size
    a = np.random.binomial(1, l*p_a_l + (1-l)*(1-p_a_l)) # Simulation of treatment
    r = np.random.normal(a*TE + np.exp(l), 2, size=(n,1)) # Simulation of recovery. The treatment effect is 4.

    # Getting them together:
    data = np.hstack((l, a, r))

    return data


###############################################################################
###       Continuous stone size simulator with gamma parametrization        ###
###############################################################################
def ks_cont_size_g_simulator(n=5000):
    """Creates data that resembles the kidney stone data set.
    
    Args:
        n: Number of observations to be generated
    Return:
        A numpy array with three columns. Size of kidney stone (if 1 Large)
          Treatment assigned (if 1 A), Recovery status (Normally distributed, depending on KS and T)
    """
    shape = 5
    scale = 2           # This is the inverse of the rate which is the way it is parametrized in pytorch

    cutoff = 10         # Cutoff for declaring big or small stones, this is the mean of the gamma
    p_a_l = 263/343     # Probability of getting treatment a given small stones
    p_b_l = 80/343      # Probability of getting treatment b given small stones

    # Simulation
    size = np.random.gamma(shape, scale, size=(n,1))        # Simulation of kidney stone size
    l = size > cutoff
    a = np.random.binomial(1, l*p_a_l + (1-l)*(1-p_a_l)) # Simulation of treatment
    r = np.random.normal(a*4 + size, 2, size=(n,1)) # Simulation of recovery. The treatment effect is 4.

    # Getting them together:
    data = np.hstack((size, a, r))

    return data


###############################################################################
###     Continuous stone size simulator with log-normal parametrization     ###
###############################################################################
def ks_cont_size_ln_simulator(n=5000):
    """Creates data that resembles the kidney stone data set.
    
    Args:
        n Number of observations to be generated
    Return:
        A numpy array with three columns. Size of kidney stone (if 1 Large)
          Treatment assigned (if 1 A), Recovery status (Normally distributed, depending on KS and T)
    """
    mu    = 2.5
    sigma = 0.25

    cutoff = 10         # Cutoff for declaring big or small stones, this is the mean of the gamma
    p_a_l = 263/343     # Probability of getting treatment a given small stones
    p_b_l = 80/343      # Probability of getting treatment b given small stones

    # Simulation
    size = np.random.lognormal(mu, sigma, size=(n,1))        # Simulation of kidney stone size
    l = size > cutoff
    a = np.random.binomial(1, l*p_a_l + (1-l)*(1-p_a_l)) # Simulation of treatment
    r = np.random.normal(a*4 + size, 2, size=(n,1)) # Simulation of recovery. The treatment effect is 4.

    # Getting them together:
    data = np.hstack((size, a, r))

    return data


###############################################################################
###                        Non-linear case simulator                        ###
###############################################################################
def ks_non_linear_simulator_logit_p(n=5000):
    """Creates data that resembles the kidney stone data set.
    
    Args:
        n: Number of observations to be generated
    Return:
        A numpy array with three columns. Size of kidney stone (continuous distributed variable)
          Treatment assigned (if 1 A), Recovery status (Normally distributed, depending on KS and T)
    """
    mu    = 2.5
    sigma = 0.25

    # Simulation
    size = np.random.lognormal(mu, sigma, size=(n,1))        # Simulation of kidney stone size
    norm_size = size-np.mean(size)
    p = 1/(1+np.exp(-norm_size/10)) # Original p = 1/(1+np.exp(-norm_size))
    a = np.random.binomial(1, p) # Simulation of treatment
    r = np.random.normal((50*a)/(size+3), 1, size=(n, 1)) # Simulation of recovery

    # Getting them together:
    data = np.hstack((size, a, r))

    return data


###############################################################################
###                             Front-door data                             ###
###############################################################################
def front_door_simulator(n):
    """Creates fake data for the front-door adjustment experiment
    """
    # Simulation
    u = np.random.normal(size=(n, 1))
    x = np.random.normal(np.sin(u), 0.1)
    z = np.random.normal(1+(-x)**2, 0.1)
    y = np.random.normal(np.sin(u**2) + 5/(z), 0.1)

    # Getting them together:
    data = np.hstack((x, z, y, u))

    return data


def main(args):
    """This function creates the data necessary to perform the experiments.
    """
    # Binary data
    data = ks_binary_simulator(args.n)
    print("Binary data created succesfully")

    # Simpson's paradox check
    binary_data_check(data)
    
    save_data(data, args.path, "ks_binary_data.npy")

    # Continuous recovery data
    data = ks_cont_recovery_simulator(args.n)
    print("Continuous treatment data created succesfully")

    save_data(data, args.path, "ks_cont_rec_data.npy")
        
    # Continuous size with gamma parametrization
    data = ks_cont_size_g_simulator(args.n)
    print("Continuous size data with gamma parametrization created succesfully")
    save_data(data, args.path, "ks_cont_size_data_g.npy")

    # Continuous size with log-normal parametrization
    data = ks_cont_size_ln_simulator(args.n)
    print("Continuous size data with log-normal parametrization created succesfully")
    save_data(data, args.path, "ks_cont_size_data_ln.npy")

    # Non-linear data
    data = ks_non_linear_simulator_logit_p(args.n)
    print("Non-linear data with logit probabilities created succesfully")
    save_data(data, args.path, "ks_non_linear_data_lp.npy")

    # Front-door data
    data = front_door_simulator(args.n)
    print("Front-door data created succesfully")
    save_data(data, args.path, "front_door_data.npy")

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, default=5000, help='Number of samples of fake data to create.')
    parser.add_argument('path', default='./', help='Path where the datasets should be saved.')
    args = parser.parse_args()

    main(args)