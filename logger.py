import logging

def get_logger(filename, local_rank):
    formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if filename is not None and local_rank <=0: # only log to file for first GPU
        f_handler = logging.FileHandler(filename, 'a')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.INFO)
        logger.addHandler(stdout_handler)
    else: # null handlers for other GPUs
        null_handler = logging.NullHandler()
        null_handler.setLevel(logging.INFO)
        logger.addHandler(null_handler)
    
    return logger
