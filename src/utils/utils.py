#! utils.py
""" Helper functions and classes
"""
import logging
import numpy as np
import os
import sys


def initialize_logger(logger_path):
    """ Helper function to initialize a logger.
    """
    logging.basicConfig(filename=logger_path,
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)

    output_file_handler = logging.FileHandler(logger_path)
    stdout_handler = logging.StreamHandler(sys.stdout)

    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)

    return logger


def get_freer_gpu():
    """ Returns the index of the NVIDIA GPU with least usage.
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove("tmp")
    
    return np.argmax(memory_available)
    