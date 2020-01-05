import concurrent
import datetime
import math
import pathlib
import random
import threading
import time
from timeit import default_timer as timer
import os

def get_object_bytes(keys, *,
                     save_root_path=None,
                     verbose=True):
    result = {}
    # TODO: parallelize this as well?
    for key in keys:
        local_filepath = save_root_path / key
        if not local_filepath.is_file():
            local_filepath.parent.mkdir(parents=True, exist_ok=True)
            print('Key %s does not exist!' % local_filepath)

        if verbose:
            print('Reading from local file {} ... '.format(local_filepath), end='')
        with open(local_filepath, 'rb') as f:
            result[key] = f.read()
        if verbose:
            print('done')
    return result


def default_option_if_needed(*, user_option, default):
    if user_option is None:
        return default
    else:
        return user_option


class LeoWrapper:
    def __init__(self,
                 save_root_path=None,
                 verbose=True,
                 max_num_threads=20,
                 num_tries=5,
                 initial_delay=1.0,
                 delay_factor=math.sqrt(2.0),
                 skip_modification_time_check=False):

        # save_root_path should be the path to "CoreRepo" for the data, is a posix path
        assert save_root_path is not None
        self.save_root_path = pathlib.Path(save_root_path).resolve()
        self.save_root_path.mkdir(parents=True, exist_ok=True)
        assert self.save_root_path.is_dir()

        self.verbose = verbose
        self.max_num_threads = max_num_threads
        self.num_tries = num_tries
        self.initial_delay = initial_delay
        self.delay_factor = delay_factor
        self.skip_modification_time_check = skip_modification_time_check
        
    # Save bytes (of pickle.dumps type) to a pkl file
    def put(self, bytes_to_store, key, verbose=None):
        cur_verbose = default_option_if_needed(user_option=verbose, default=self.verbose)
        complete_path = self.save_root_path / key
        complete_path.parent.mkdir(parents=True, exist_ok=True)

        file = open(complete_path, 'wb')
        file.write(bytes_to_store)
        file.close()
        if cur_verbose:
            print('Stored {} bytes under key {}'.format(len(bytes_to_store), key))
    
    def put_multiple(self, data, verbose=None, callback=None):
        for key, bytes_to_store in data.items():
            self.put(bytes_to_store, key, verbose)

            
    # Load bytes from pkl file
    def get(self, key, verbose=None, skip_modification_time_check=None):
        return self.get_multiple([key], verbose=verbose, skip_modification_time_check=skip_modification_time_check)[key]
    
    def get_multiple(self, keys, verbose=None, callback=None, skip_modification_time_check=None):
        if verbose is None:
            cur_verbose = self.verbose
        else:
            cur_verbose = verbose
        cur_verbose = default_option_if_needed(user_option=verbose, default=self.verbose)
        return get_object_bytes(keys, 
                                save_root_path=self.save_root_path,
                                verbose=cur_verbose)
