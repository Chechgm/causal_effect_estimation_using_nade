#! ./data/fake_data.py
""" Creates a fake dataset that resembles that of the kideny stone example.

The available functions are:
- save_data
- binary_simulator
- binary_data_check
- continuous_outcome_simulator
- continuous_confounder_gamma_simulator
- continuous_confounder_logn_simulator
- non_linear_simulator
- unobserved_confounder_mild_simulator
- unobserved_confounder_strong_simulator
- front_door_simulator
- main
"""
import argparse
import logging
import numpy as np
import os.path


def save_data(data, path, file_name, logger):
    """Saves the data in the desired path, if it doesn't exist.
    """
    if not os.path.exists(os.path.join(path, file_name)):
        np.save(os.path.join(path, file_name), data)
        logger.info(f"{file_name} saved succesfully")
    else:
        logger.info(f"{file_name} already existed")


###############################################################################
#            Binary data simulator and Simpson's paradox "Checker"            #
###############################################################################
def binary_simulator(n=5000):
    """Creates data that resembles the kidney stone data set.

    Args:
        n: Number of observations to be generated
    Return:
        A numpy array with three columns. Size of kidney stone (if 1 Large)
          Treatment assigned (if 1 A), Recovery status (if 1 recovered)
    """
    p_l = 343/(343+357)  # Probability of large kidney stones
    p_a_s = 263/343  # Probability of getting treatment a given small stones
    p_a_l = 87/357  # Probability of treatment a given large stones
    p_r_l_a = 192/263  # Probability of recovery given large stones and treatment a
    p_r_l_b = 55/80  # Probability of recovery given large stones and treatement b
    p_r_s_a = 81/87  # Probability of recovery given small stones and treatment a
    p_r_s_b = 234/270  # Probability of recovery given small stones and treatement b

    # Simulation
    large = np.random.binomial(1, p_l, size=(n,1))  # Simulation of kidney stone size
    a = np.random.binomial(1, large*p_a_l + (1-large)*p_a_s)  # Simulation of treatment
    r = np.random.binomial(1, large*a*p_r_l_a + large*(1-a)*p_r_l_b + (1-large)*a*p_r_s_a + (1-large)*(1-a)*p_r_s_b)

    # Getting them together:
    data = np.hstack((large, a, r))

    return data


# Simpson's paradox checker
def binary_data_check(data, logger):
    """
    This function checks whether the binary data generated follows Simpson's paradox.
    """
    # Check that P(R=1 | B) > P(R=1 | A)
    p_r_b = np.mean(data[data[:,1]==False, 2])  # P(R=1 | B)
    p_r_a = np.mean(data[data[:,1]==True, 2])  # P(R=1 | A)
    logger.info(f"is P(R=1 | B) > P(R=1 | A)? {p_r_b > p_r_a}")

    # Check that, individually, P(R=1 | A, S) > P(R=1 | B, S)
    #   and P(R=1 | A, L) > P(R=1 | B, L)

    mask_l_a = ((data[:,1]==True) & (data[:,0]==True))
    p_r_l_a = np.mean(data[mask_l_a, 2])  # P(R=1 | A, L)
    mask_l_b = ((data[:,1]==False) & (data[:,0]==True))
    p_r_l_b = np.mean(data[mask_l_b, 2])  # P(R=1 | B, L)
    logger.info(f"is P(R=1 | A, L) > P(R=1 | B, L)? {p_r_l_a > p_r_l_b}")

    mask_s_a = ((data[:,1]==True) & (data[:,0]==False))
    p_r_s_a = np.mean(data[mask_s_a, 2])  # P(R=1 | A, S)
    mask_s_b = ((data[:,1]==False) & (data[:,0]==False))
    p_r_s_b = np.mean(data[mask_s_b, 2])  # P(R=1 | B, S)
    logger.info(f"is P(R=1 | A, S) > P(R=1 | B, S)? {p_r_s_a > p_r_s_b}")


###############################################################################
#                        Continuous recovery simulator                        #
###############################################################################
def continuous_outcome_simulator(n):
    """Creates data that resembles the kidney stone data set.

    Args:
        n: Number of observations to be generated
    Output:
        A numpy array with three columns. Size of kidney stone (if 1 Large)
          Treatment assigned (if 1 A), Recovery status (Normally distributed, depending on KS and T)
    """
    p_l = 343/(343+357)  # Probability of large kidney stones
    p_a_s = 263/343  # Probability of getting treatment a given small stones
    p_a_l = 87/357  # Probability of treatment a given large stones
    TE = 4  # Treatment effect

    # Simulation
    large = np.random.binomial(1, p_l, size=(n,1))  # Simulation of kidney stone size
    a = np.random.binomial(1, large*p_a_l + (1-large)*p_a_s)  # Simulation of treatment
    r = np.random.normal(a*TE + np.exp(large), 2, size=(n,1))  # Simulation of recovery. The treatment effect is 4.

    # Getting them together:
    data = np.hstack((large, a, r))

    return data


###############################################################################
#         Continuous confounder simulator with gamma parametrization          #
###############################################################################
def continuous_confounder_gamma_simulator(n=5000):
    """Creates data that resembles the kidney stone data set.

    Args:
        n: Number of observations to be generated
    Return:
        A numpy array with three columns. Size of kidney stone (if 1 Large)
          Treatment assigned (if 1 A), Recovery status (Normally distributed, depending on KS and T)
    """
    shape = 5
    scale = 2  # This is the inverse of the rate which is the way it is parametrized in pytorch

    cutoff = 10  # Cutoff for declaring big or small stones, this is the mean of the gamma
    p_a_s = 263/343  # Probability of getting treatment a given small stones
    p_a_l = 87/357  # Probability of treatment a given large stones

    # Simulation
    size = np.random.gamma(shape, scale, size=(n,1))  # Simulation of kidney stone size
    large = size > cutoff
    a = np.random.binomial(1, large*p_a_l + (1-large)*p_a_s)  # Simulation of treatment
    r = np.random.normal(a*4 + size, 1, size=(n,1))  # Simulation of recovery. The treatment effect is 4.

    # Getting them together:
    data = np.hstack((size, a, r))

    return data


###############################################################################
#       Continuous stone size simulator with log-normal parametrization       #
###############################################################################
def continuous_confounder_logn_simulator(n=5000):
    """Creates data that resembles the kidney stone data set.

    Args:
        n Number of observations to be generated
    Return:
        A numpy array with three columns. Size of kidney stone (if 1 Large)
          Treatment assigned (if 1 A), Recovery status (Normally distributed, depending on KS and T)
    """
    mu = 2.5
    sigma = 0.25

    cutoff = 10  # Cutoff for declaring big or small stones, this is the mean of the gamma
    p_a_s = 263/343  # Probability of getting treatment a given small stones
    p_a_l = 87/357  # Probability of treatment a given large stones

    # Simulation
    size = np.random.lognormal(mu, sigma, size=(n,1))  # Simulation of kidney stone size
    large = size > cutoff
    a = np.random.binomial(1, large*p_a_l + (1-large)*p_a_s)  # Simulation of treatment
    r = np.random.normal(a*4 + size, 1, size=(n,1))  # Simulation of recovery. The treatment effect is 4.

    # Getting them together:
    data = np.hstack((size, a, r))

    return data


###############################################################################
#                          Non-linear case simulator                          #
###############################################################################
def non_linear_simulator(n=5000):
    """Creates data that resembles the kidney stone data set.

    Args:
        n: Number of observations to be generated
    Return:
        A numpy array with three columns. Size of kidney stone (continuous distributed variable)
          Treatment assigned (if 1 A), Recovery status (Normally distributed, depending on KS and T)
    """
    mu = 2.5
    sigma = 0.25

    # Simulation
    size = np.random.lognormal(mu, sigma, size=(n,1))  # Simulation of kidney stone size
    norm_size = size-np.mean(size)
    p = 1/(1+np.exp(-norm_size/10))  # Original p = 1/(1+np.exp(-norm_size))
    a = np.random.binomial(1, p)  # Simulation of treatment
    r = np.random.normal((50*a)/(size+3), 1, size=(n, 1))  # Simulation of recovery

    # Getting them together:
    data = np.hstack((size, a, r))

    return data


###############################################################################
#                      Unobserved confounder simulators                       #
###############################################################################
def unobserved_confounder_mild_simulator(n=5000):
    """Creates data that resembles the kidney stone data set.

    Args:
        n: Number of observations to be generated
    Return:
        A numpy array with three columns. Size of kidney stone (continuous distributed variable)
          Treatment assigned (if 1 A), Recovery status (Normally distributed, depending on KS and T)
    """
    mu = 2.5
    sigma = 0.25

    # Simulation
    size = np.random.lognormal(mu, sigma, size=(n,1))  # Simulation of kidney stone size
    norm_size = size-np.mean(size)
    confounder = np.random.normal(0, 1, size=(n,1))  # Simulation of kidney stone size
    norm_confounder = size-np.mean(confounder)
    p = 1/(1+np.exp(-norm_size-norm_confounder/10))
    a = np.random.binomial(1, p)  # Simulation of treatment
    r = np.random.normal((50*a)/(size+3)+0.3*norm_confounder, 1, size=(n, 1))  # Simulation of recovery

    # Getting them together:
    data = np.hstack((size, a, r))

    return data


def unobserved_confounder_strong_simulator(n=5000):
    """Creates data that resembles the kidney stone data set.

    Args:
        n: Number of observations to be generated
    Return:
        A numpy array with three columns. Size of kidney stone (continuous distributed variable)
          Treatment assigned (if 1 A), Recovery status (Normally distributed, depending on KS and T)
    """
    mu = 2.5
    sigma = 0.25

    # Simulation
    size = np.random.lognormal(mu, sigma, size=(n,1))  # Simulation of kidney stone size
    norm_size = size-np.mean(size)
    confounder = np.random.normal(0, 1, size=(n,1))  
    norm_confounder = size-np.mean(confounder)
    p = 1/(1+np.exp(-norm_size-norm_confounder/10))
    a = np.random.binomial(1, p)  # Simulation of treatment
    r = np.random.normal((50*a)/(size+3)+3*norm_confounder, 1, size=(n, 1))  # Simulation of recovery

    # Getting them together:
    data = np.hstack((size, a, r))

    return data


###############################################################################
#                               Front-door data                               #
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
    """ Creates the data necessary to perform the experiments.
    """
    # Data simulation logger set-up
    logger_filename = os.path.join(args.path, "data_simulation_logger.log")
    logging.basicConfig(filename=logger_filename,
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Binary data
    data = binary_simulator(args.n)
    logger.info("Binary data created succesfully")

    # Simpson's paradox check
    binary_data_check(data, logger)

    save_data(data, args.path, "binary_data.npy", logger)

    # Continuous recovery data
    data = continuous_outcome_simulator(args.n)
    logger.info("Continuous treatment data created succesfully")

    save_data(data, args.path, "continuous_outcome_data.npy", logger)

    # Continuous size with gamma parametrization
    data = continuous_confounder_gamma_simulator(args.n)
    logger.info("Continuous size data with gamma parametrization created succesfully")
    save_data(data, args.path, "continuous_confounder_gamma_data.npy", logger)

    # Continuous size with log-normal parametrization
    data = continuous_confounder_logn_simulator(args.n)
    logger.info("Continuous size data with log-normal parametrization created succesfully")
    save_data(data, args.path, "continuous_confounder_logn_data.npy", logger)

    # Non-linear data
    data = non_linear_simulator(args.n)
    logger.info("Non-linear data created succesfully")
    save_data(data, args.path, "non_linear_data.npy", logger)

    # Unobserved confounder mild
    data = unobserved_confounder_mild_simulator(args.n)
    logger.info("Unobserved confounder mild data created succesfully")
    save_data(data, args.path, "unobserved_confounder_mild_data.npy", logger)

    # Unobserved confounder strong
    data = unobserved_confounder_strong_simulator(args.n)
    logger.info("Unobserved confounder strong data created succesfully")
    save_data(data, args.path, "unobserved_confounder_strong_data.npy", logger)

    # Front-door data
    data = front_door_simulator(args.n)
    logger.info("Front-door data created succesfully")
    save_data(data, args.path, "front_door_data.npy", logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5000, help='Number of samples of fake data to create.')
    parser.add_argument('--path', default='./', help='Path where the datasets should be saved.')
    args = parser.parse_args()

    main(args)
