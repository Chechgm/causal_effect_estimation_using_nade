# This script creates a fake dataset that resembles that of the kideny stone example
import numpy  as np
import os.path

###############################################################################
###         Binary data simulator and Simpson's paradox "Checker"           ###
###############################################################################
def ks_binary_simulator(n=5000):
    """
    Creates data that resembles the kidney stone data set.
    Inputs:
    n Number of observations to be generated
    Output:
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
    This function checks that the binary data generated follows the Simpson's paradox.
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

# Create the data
data = ks_binary_simulator()
print("Binary data created succesfully")

# Check the data
binary_data_check(data)

# Saving them if not saved already
if not os.path.exists("./ks_binary_data.npy"):
    np.save("./ks_binary_data.npy", data)
    print("Binary data saved succesfully")

###############################################################################
###                     Continuous recovery simulator                       ###
###############################################################################
def ks_cont_recovery_simulator(n=5000):
    """
    Creates data that resembles the kidney stone data set.
    Inputs:
    n Number of observations to be generated
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
    r = np.random.normal(TE*4 + np.exp(l), 2, size=(n,1)) # Simulation of recovery. The treatment effect is 4.

    # Getting them together:
    data = np.hstack((l, a, r))

    return data

data = ks_cont_recovery_simulator()
print("Continuous treatment data created succesfully")

# Saving them if not saved already
if not os.path.exists("./ks_cont_rec_data.npy"):
    np.save("./ks_cont_rec_data.npy", data)
    print("Continuous treatment data saved succesfully")

###############################################################################
###                     Continuous stone size simulator                     ###
###############################################################################
def ks_cont_size_simulator(n=5000):
    """
    Creates data that resembles the kidney stone data set.
    Inputs:
    n Number of observations to be generated
    Output:
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
    r = np.random.normal(a*4 + 1/size, 2, size=(n,1)) # Simulation of recovery. The treatment effect is 4.

    # Getting them together:
    data = np.hstack((size, a, r))

    return data

data = ks_cont_size_simulator()
print("Continuous size data created succesfully")

# Saving them if not saved already
if not os.path.exists("./ks_cont_size_data.npy"):
    np.save("./ks_cont_size_data.npy", data)
    print("Continuous size data saved succesfully")
