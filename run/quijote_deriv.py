'''

script for calculating derivatives of B w.r.t. theta (dB/dtheta) of Paco's quijote sims 
and saving them into easy to access files

'''
import os 
import glob
import numpy as np 
# -- emanu --
from emanu import util as UT
from emanu import forecast as Forecast


def dBdtheta(theta, z=0, rsd=True, flag=None): 
    ''' write out derivatives of B with respect to theta to file in the following format: 
    k1, k2, k3, dB/dtheta, dlogB/dtheta.

    The fiducial derivatives include all B calculations: n-body, fixed-paired, different rsd 
    directions. If theta == 'Mnu', it will output all the different methods for calculating 
    derivatives: 'fin', 'fin0', 'p', 'pp', 'ppp'. 
   
    :param theta: 
        parameter value

    :param z: (default: 0) 
        redshift. currently only z=0 is implemented

    '''
    fderiv = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.%s%s%s.dat' % (theta, _rsd_str(rsd), _flag_str(flag))) 
    print("--- writing dB/d%s to %s ---" % (theta, fderiv))

    if theta == 'Mnu': 
        for i_dmnu, dmnu in enumerate(['fin', 'fin0', 'p', 'pp', 'ppp']): 
            # calculate dB/dtheta and dlogB/dtheta 
            k1, k2, k3, dbk = Forecast.quijote_dBkdtheta(theta, log=False, z=z, dmnu=dmnu, flag=flag, rsd=rsd, Nfp=None)
            _, _, _, dlogbk = Forecast.quijote_dBkdtheta(theta, log=True, z=z, dmnu=dmnu, flag=flag, rsd=rsd, Nfp=None)
            
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
        k1, k2, k3, dbk = Forecast.quijote_dBkdtheta(theta, log=False, z=z, flag=flag, rsd=rsd, Nfp=None)
        _, _, _, dlogbk = Forecast.quijote_dBkdtheta(theta, log=True, z=z, flag=flag, rsd=rsd, Nfp=None)

        hdr = 'k1, k2, k3, dB/dtheta, dlogB/dtheta'
        # save to file 
        np.savetxt(fderiv, np.vstack([k1, k2, k3, dbk, dlogbk]).T, header=hdr, delimiter=',\t', fmt='%i %i %i %.5e %.5e')
    return None 


def dPdtheta(theta, z=0): 
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
    fderiv = os.path.join(UT.doc_dir(), 'dat', 'dPdtheta.%s.dat' % theta) 
    print("--- writing dP/d%s to %s ---" % (theta, fderiv))

    if theta == 'Mnu': 
        for i_dmnu, dmnu in enumerate(['fin', 'fin0', 'p', 'pp', 'ppp']): 
            # calculate dP/dtheta and dlogP/dtheta 
            k, dpk = Forecast.quijote_dPkdtheta(theta, log=False, z=z, dmnu=dmnu, Nfp=None)
            _, dlogpk = Forecast.quijote_dPkdtheta(theta, log=True, z=z, dmnu=dmnu, Nfp=None)
            
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
        k, dpk = Forecast.quijote_dPkdtheta(theta, log=False, z=z, Nfp=None)
        _, dlogpk = Forecast.quijote_dPkdtheta(theta, log=True, z=z, Nfp=None)

        hdr = 'k, dP/dtheta, dlogP/dtheta'
        # save to file 
        np.savetxt(fderiv, np.vstack([k, dpk, dlogpk]).T, header=hdr, delimiter=',\t', fmt='%.5e %.5e %.5e')
    return None 


def _rsd_str(rsd): 
    # assign string based on rsd kwarg 
    if type(rsd) == bool: return ''
    elif type(rsd) == int: return '.rsd%i' % rsd
    elif rsd == 'real': return '.real'


def _flag_str(flag): 
    # assign string based on flag kwarg
    return ['.%s' % flag, ''][flag is None]


if __name__=="__main__": 
    thetas = ['Mnu', 'Om', 'Ob2', 'h', 'ns', 's8', 'Mmin', 'Amp', 'Asn', 'Bsn', 'b2', 'g2']
    for theta in thetas: 
        continue
        dBdtheta(theta, z=0)
        dBdtheta(theta, z=0, rsd='real')
        for rsd in [True, 0, 'real']: 
            dBdtheta(theta, z=0, rsd=rsd, flag='ncv')
            dBdtheta(theta, z=0, rsd=rsd, flag='reg') 

    thetas = ['Mnu', 'Om', 'Ob', 'h', 'ns', 's8', 'Mmin', 'Amp', 'Asn']
    for theta in thetas: 
        dPdtheta(theta, z=0)
