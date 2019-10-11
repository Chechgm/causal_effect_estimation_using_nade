# This script creates a fake dataset that resembles that of the kideny stone example
import numpy  as np
import os.path

def kidney_stones_data_generator(n=1000):
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
    p_r_l_a = 192/263   # Probability of recovery given small stones and treatment a
    p_r_l_b = 55/80     # Probability of recovery given small stones and treatement b
    p_r_s_a = 81/87     # Probability of recovery given small stones and treatment a
    p_r_s_b = 234/270   # Probability of recovery given small stones and treatement b

    # Simulation
    l = np.random.binomial(1, p_l, size=(n,1))           # Simulation of kidney stone size
    a = np.random.binomial(1, l*p_a_l + (1-l)*(1-p_a_l)) # Simulation of treatment
    r = np.random.binomial(1, l*a*p_r_l_a + l*(1-a)*p_r_l_b + (1-l)*a*p_r_s_a + (1-l)*(1-a)*p_r_s_b)

    # Getting them together:
    data = np.hstack((l, a, r))

    return data

data = kidney_stones_data_generator()
print("Data created succesfully")

# Saving them if not saved already
if not os.path.exists('./data'):
    os.mkdir('./data')
    np.save('./data/kidney_data.npy', data)
    print("Data saved succesfully")
elif not os.path.exists('./data/kidney_data.npy'):
    np.save('./data/kidney_data.npy', data)
    print("Data saved succesfully")

# # Sanity checks (means of the recovery column)
# np.mean(data[data[:,1]==False, 2]) # Treatment B should be greater than Treatment A
# np.mean(data[data[:,1]==True, 2])  # Treatment A
#
#
# mask_l_a = ((data[:,1]==True) & (data[:,0]==True))
# np.mean(data[mask_l_a, 2])  # Treatment A and Large stones, should be greater than Treatment B and Large stones
# mask_l_b = ((data[:,1]==False) & (data[:,0]==True))
# np.mean(data[mask_l_b, 2])  # Treatment B and Large stones
#
# mask_s_a = ((data[:,1]==True) & (data[:,0]==False))
# np.mean(data[mask_s_a, 2])  # Treatment A and small stones, should be greater than Treatment B and Small stones
# mask_s_b = ((data[:,1]==False) & (data[:,0]==False))
# np.mean(data[mask_s_b, 2])  # Treatment B and Small stones
