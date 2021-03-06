'''

utility functions 

'''
import os
import sys
import numpy as np


def ijl_order(i_k, j_k, l_k, typ='GM'): 
    ''' triangle configuration ordering, returns indices
    '''
    i_bq = np.arange(len(i_k))
    if typ == 'GM': # same order as Hector's
        i_bq_new = [] 

        l_usort = np.sort(np.unique(l_k))
        for l in l_usort: 
            j_usort = np.sort(np.unique(j_k[l_k == l]))
            for j in j_usort: 
                i_usort = np.sort(np.unique(i_k[(l_k == l) & (j_k == j)]))
                for i in i_usort: 
                    i_bq_new.append(i_bq[(i_k == i) & (j_k == j) & (l_k == l)])
    else: 
        raise NotImplementedError
    return np.array(i_bq_new).flatten()


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
    return os.path.join(code_dir(), 'figs')


def doc_dir(): 
    ''' directory for paper related stuff 
    '''
    return os.path.join(code_dir(), 'doc')


def hades_dir(mneut, nreal): 
    ''' directory of hades data 
    '''
    if mneut == 0.1: 
        _dir = ''.join([dat_dir(), '0.10eV/', str(nreal), '/'])
    else: 
        _dir = ''.join([dat_dir(), str(mneut), 'eV/', str(nreal), '/'])
    return _dir


def fig_tex(ffig, pdf=False): 
    ''' given filename of figure return a latex friendly file name
    '''
    path, ffig_base = os.path.split(ffig) 
    ext = ffig_base.rsplit('.', 1)[-1] 
    ffig_name = ffig_base.rsplit('.', 1)[0]

    _ffig_name = ffig_name.replace('.', '_') 
    if pdf: ext = 'pdf' 
    return os.path.join(path, '.'.join([_ffig_name, ext])) 

