'''

utility functions 

'''
import os
import sys
import numpy as np


def check_env(): 
    if os.environ.get('EMANU_DIR') is None: 
        raise ValueError("set $EMANU_DIR in bashrc file!") 
    return None


def dat_dir(): 
    return os.environ.get('EMANU_DIR') 


def code_dir(): 
    return os.environ.get('EMANU_CODEDIR') 


def fig_dir(): 
    ''' directory for figures 
    '''
    return code_dir()+'figs/'


def doc_dir(): 
    ''' directory for paper related stuff 
    '''
    return code_dir()+'doc/'


def hades_dir(mneut, nreal): 
    ''' directory of hades data 
    '''
    if mneut == 0.1: 
        _dir = ''.join([dat_dir(), '0.10eV/', str(nreal), '/'])
    else: 
        _dir = ''.join([dat_dir(), str(mneut), 'eV/', str(nreal), '/'])
    return _dir
