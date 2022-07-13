import time
import logging

import torch.cuda
import langcodes

def get_logger( class_name, verbose = True ):
    logger = logging.getLogger( f'{class_name}' )

    logger.propagate = True

    if not logger.hasHandlers():
        log_handler = logging.StreamHandler()
        log_formatter = logging.Formatter(fmt="[%(levelname)s/%(processName)s/%(name)s] %(message)s",
                              datefmt='%Y-%m-%d %H:%M:%S')
        log_handler.setFormatter(log_formatter)

        logger.addHandler(log_handler)

    level = logging.INFO if verbose else logging.WARNING
    logger.setLevel( level )

    return logger

class Timer:
    def __init__(self, text = '', logger = None):
        self.text = text
        self.logger = logger

    def __enter__(self):
        self.start = time.perf_counter()
        if self.logger:
            self.logger.info( f'Start: "{self.text}"' )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end = time.perf_counter()
        if self.logger:
            self.logger.info( f'Completed: "{self.text}" in {(end - self.start):.04f}s')

    @property
    def time( self ):
        curr_time = time.perf_counter()
        return curr_time - self.start

# Wrapper for langcodes.closest_supported_match, with additional check that
# all strings in candidate_list are in fact languages
# https://github.com/rspeer/langcodes#finding-the-best-matching-language
def find_closest_language( language, candidate_list, max_distance = 10 ):
    if not langcodes.tag_is_valid(language):
        return None
    candidate_list = [lang for lang in candidate_list if langcodes.tag_is_valid(lang)]
    return langcodes.closest_supported_match( language, candidate_list, max_distance )

def log_cuda_info(logger = None):
    if torch.cuda.is_available() and logger:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        logger.info(f'Total GPU memory: {t/2**30:.6} GiB, reserved: {r/2**30:.6} GiB, allocated: {a/2**30:.6} GiB, free: {f/2**30:.6} GiB')

# Quickest way to count number of lines in file according to this:
# https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python/68385697#68385697
def buf_count_newlines_gen(fname):
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count