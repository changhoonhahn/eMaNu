'''

script for calculating derivatives of B w.r.t. theta (dB/dtheta) of the quijote
HOD galaxy catalogs and saving them into easy to access files. 

'''
import os 
import glob
import numpy as np 
# -- emanu --
from emanu import util as UT
from emanu import forecast as Forecast


def hod_dBdtheta(theta, z=0, rsd='all', seed=0, flag=None, silent=True): 
    ''' write out derivatives of HOD galaxy B with respect to theta to file 
    in the following format: 
    # k1, k2, k3, dB/dtheta, dlogB/dtheta.

    The fiducial derivatives include all B calculations: n-body, fixed-paired, different rsd 
    directions. If theta == 'Mnu', it will output all the different methods for calculating 
    derivatives: 'fin', 'fin0', 'p', 'pp', 'ppp'. 
   
    :param theta: 
        parameter value

    :param z: (default: 0) 
        redshift. currently only z=0 is implemented

    '''
    fderiv = os.path.join(UT.doc_dir(), 'dat', 
            'hod_dBdtheta.%s%s%s%s.dat' % (theta, _rsd_str(rsd), _seed_str(seed), _flag_str(flag))) 
    print("--- writing dB/d%s to %s ---" % (theta, fderiv))

    if theta == 'Mnu': 
        for i_dmnu, dmnu in enumerate(['fin', 'fin0', 'p', 'pp', 'ppp']): 
            # calculate dB/dtheta and dlogB/dtheta 
            k1, k2, k3, dbk = Forecast.quijhod_dBkdtheta(theta, 
                    log=False, z=z, dmnu=dmnu, flag=flag, rsd=rsd, seed=seed, Nderiv=None, silent=silent)
            _, _, _, dlogbk = Forecast.quijhod_dBkdtheta(theta, 
                    log=True, z=z, dmnu=dmnu, flag=flag, rsd=rsd, seed=seed, Nderiv=None, silent=silent)
            if i_dmnu == 0: 
                datastack = [k1, k2, k3]
                hdr = 'k1, k2, k3'
                fmt = '%i %i %i'
            datastack.append(dbk)
            datastack.append(dlogbk) 
            hdr += (', dB/dtheta %s, dlogB/dtheta %s' % (dmnu, dmnu)) # header
            fmt += ' %.5e %.5e' # format 
        
        np.savetxt(fderiv, np.vstack(datastack).T, header=hdr, delimiter=',\t', fmt=fmt)
    else: 
        # calculate dB/dtheta and dlogB/dtheta 
        k1, k2, k3, dbk = Forecast.quijhod_dBkdtheta(theta, 
                log=False, z=z, flag=flag, rsd=rsd, seed=seed, Nderiv=None, silent=silent)
        _, _, _, dlogbk = Forecast.quijhod_dBkdtheta(theta, 
                log=True, z=z, flag=flag, rsd=rsd, seed=seed, Nderiv=None, silent=silent)

        hdr = 'k1, k2, k3, dB/dtheta, dlogB/dtheta'
        # save to file 
        np.savetxt(fderiv, np.vstack([k1, k2, k3, dbk, dlogbk]).T, header=hdr, delimiter=',\t', fmt='%i %i %i %.5e %.5e')
    return None 


def hod_dPdtheta(theta, z=0, rsd='all', flag=None, silent=True): 
    ''' write out  derivatives of HOD galaxy P with respect to theta to file in the following format: 
    k, dP/dtheta, dlogP/dtheta.

    The fiducial derivatives include all B calculations: n-body, fixed-paired, different rsd 
    directions. If theta == 'Mnu', it will output all the different methods for calculating 
    derivatives: 'fin', 'fin0', 'p', 'pp', 'ppp'. 
   
    :param theta: 
        parameter value

    :param z: (default: 0) 
        redshift. currently only z=0 is implemented

    '''
    fderiv = os.path.join(UT.doc_dir(), 'dat', 'hod_dPdtheta.%s%s%s.dat' % (theta, _rsd_str(rsd), _flag_str(flag))) 
    print("--- writing dP/d%s to %s ---" % (theta, fderiv))

    if theta == 'Mnu': 
        for i_dmnu, dmnu in enumerate(['fin', 'fin0', 'p', 'pp', 'ppp', 'fin_2lpt']): 
            # calculate dP/dtheta and dlogP/dtheta 
            k, dpk = Forecast.quijhod_dPkdtheta(theta,
                    log=False, z=z, rsd=rsd, flag=flag, dmnu=dmnu, Nderiv=None, silent=silent)
            _, dlogpk = Forecast.quijhod_dPkdtheta(theta,
                    log=True, z=z, rsd=rsd, flag=flag, dmnu=dmnu, Nderiv=None, silent=silent)
            
            if i_dmnu == 0: 
                datastack = [k]
                hdr = 'k,'
                fmt = '%.5e'
            datastack.append(dpk)
            datastack.append(dlogpk) 
            hdr += (', dP/dtheta %s, dlogP/dtheta %s' % (dmnu, dmnu)) # header
            fmt += ' %.5e %.5e' # format 
        
        np.savetxt(fderiv, np.vstack(datastack).T, header=hdr, delimiter=',\t', fmt=fmt)
    else: 
        # calculate dP/dtheta and dlogP/dtheta 
        k, dpk = Forecast.quijhod_dPkdtheta(theta, 
                log=False, z=z, rsd=rsd, flag=flag, Nderiv=None, silent=silent)
        _, dlogpk = Forecast.quijhod_dPkdtheta(theta, 
                log=True, z=z, rsd=rsd, flag=flag, Nderiv=None, silent=silent)

        hdr = 'k, dP/dtheta, dlogP/dtheta'
        # save to file 
        np.savetxt(fderiv, np.vstack([k, dpk, dlogpk]).T, header=hdr, delimiter=',\t', fmt='%.5e %.5e %.5e')
    return None 


def hod_dP02dtheta(theta, z=0, rsd='all', seed=0, flag=None, silent=True): 
    ''' write out  derivatives of P with respect to theta to file in the following format: 
    k, dP/dtheta, dlogP/dtheta.

    The fiducial derivatives include all B calculations: n-body, fixed-paired, different rsd 
    directions. If theta == 'Mnu', it will output all the different methods for calculating 
    derivatives: 'fin', 'fin0', 'p', 'pp', 'ppp'. 
   
    :param theta: 
        parameter value

    :param z: (default: 0) 
        redshift. currently only z=0 is implemented

    '''
    fderiv = os.path.join(UT.doc_dir(), 'dat', 
            'hod_dP02dtheta.%s%s%s%s.dat' % (theta, _rsd_str(rsd), _seed_str(seed), _flag_str(flag))) 
    print("--- writing dP02/d%s to %s ---" % (theta, fderiv))

    if theta == 'Mnu': 
        for i_dmnu, dmnu in enumerate(['fin', 'fin0', 'p', 'pp', 'ppp']): 
            # calculate dP/dtheta and dlogP/dtheta 
            k, dpk = Forecast.quijhod_dP02kdtheta(theta, 
                    log=False, z=z, rsd=rsd, seed=seed, flag=flag, dmnu=dmnu, Nderiv=None, silent=silent)
            _, dlogpk = Forecast.quijhod_dP02kdtheta(theta, 
                    log=True, z=z, rsd=rsd, seed=seed, flag=flag, dmnu=dmnu, Nderiv=None, silent=silent)
            
            if i_dmnu == 0: 
                datastack = [k]
                hdr = 'k,'
                fmt = '%.5e'
            datastack.append(dpk)
            datastack.append(dlogpk) 
            hdr += (', dP02/dtheta %s, dlogP02/dtheta %s' % (dmnu, dmnu)) # header
            fmt += ' %.5e %.5e' # format 
        
        np.savetxt(fderiv, np.vstack(datastack).T, header=hdr, delimiter=',\t', fmt=fmt)
    else: 
        # calculate dP/dtheta and dlogP/dtheta 
        k, dpk = Forecast.quijhod_dP02kdtheta(theta,
                log=False, z=z, rsd=rsd, seed=seed, flag=flag, Nderiv=None, silent=silent)
        _, dlogpk = Forecast.quijhod_dP02kdtheta(theta, 
                log=True, z=z, rsd=rsd, seed=seed, flag=flag, Nderiv=None, silent=silent)

        hdr = 'k, dP02/dtheta, dlogP02/dtheta'
        # save to file 
        np.savetxt(fderiv, np.vstack([k, dpk, dlogpk]).T, header=hdr, delimiter=',\t', fmt='%.5e %.5e %.5e')
    return None 


def _rsd_str(rsd): 
    # assign string based on rsd kwarg 
    if rsd == 'all': return ''
    elif rsd in [0, 1, 2]: return '.rsd%i' % rsd
    elif rsd == 'real': return '.real'
    else: raise NotImplementedError


def _seed_str(seed): 
    if isinstance(seed, int): 
        return '.seed%i' % seed
    else: 
        if np.array_equal(np.array(seed), np.arange(np.min(seed), np.max(seed)+1)): 
            return '.seed%ito%i' % (np.min(seed), np.max(seed))
        else: 
            return '.seed%s' % '_'.join([str(s) for s in seed])


def _flag_str(flag): 
    # assign string based on flag kwarg
    return ['.%s' % flag, ''][flag is None]


if __name__=="__main__": 
    #thetas = ['Mnu', 'Om', 'Ob2', 'h', 'ns', 's8', 'logMmin', 'sigma_logM', 'logM0', 'alpha', 'logM1'] 
    #thetas = ['Asn', 'Bsn']
    thetas = ['sigma_logM_HR']
    #thetas = ['b1']
    for theta in thetas: 
        for rsd in [0, 1, 2, 'all']: # 'real', 
            for flag in ['reg']: #'ncv' 
                for seed in [0]:#, range(5)]: 
                    if theta not in ['Bsn']: 
                        #hod_dPdtheta(theta, z=0, rsd=rsd, flag=flag, silent=False)
                        if rsd != 'real': hod_dP02dtheta(theta, z=0, rsd=rsd, seed=seed, flag=flag, silent=False)
                    hod_dBdtheta(theta, z=0, rsd=rsd, seed=seed, flag=flag, silent=False) 
