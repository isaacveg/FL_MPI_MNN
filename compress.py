import os
import MNN.expr as F
import time

def param_compress(model, action='top_k', extra_info=''):
    """
    Support multiple compress functions.
    ## top_k: 
    select top_k parameters, 
    'extra_info': the compress ratio how many params remained
    returns time consumed and params in form of dict.

    ## random
    randomly select parameters, 
    'extra_info': the compress ratio how many params remained
    returns time consumed and params in form of dict.

    ## to_int8
    Naively turn parameters from float to int8. 
    'extra_info': unsaturated or saturated
    returns time consumed and params in form of dict
    ATTENTION: use 'decompress' to turn it back to fp32 before trainning!
    """
    t_0 = time.time()
    if action == 'top_k':
        
        
    
