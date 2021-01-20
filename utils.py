#! utils.py
""" Helper functions and classes
"""
import logging
import sys


# Logger set-up
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