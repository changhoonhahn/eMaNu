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


def dBdtheta_fiducial(theta, z=0): 
    ''' write out fiducial derivatives of B with respect to theta to file in the following format: 
    k1, k2, k3, dB/dtheta, dlogB/dtheta.

    The fiducial derivatives include all B calculations: n-body, fixed-paired, different rsd 
    directions. If theta == 'Mnu', it will output all the different methods for calculating 
    derivatives: 'fin', 'fin0', 'p', 'pp', 'ppp'. 
   
    :param theta: 
        parameter value

    :param z: (default: 0) 
        redshift. currently only z=0 is implemented

    '''
    fderiv = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.%s.dat' % theta) 
    print("--- writing dB/d%s to %s ---" % (theta, fderiv))

    if theta == 'Mnu': 
        for i_dmnu, dmnu in enumerate(['fin', 'fin0', 'p', 'pp', 'ppp']): 
            # calculate dB/dtheta and dlogB/dtheta 
            k1, k2, k3, dbk = Forecast.quijote_dBkdtheta(theta, log=False, z=z, dmnu=dmnu, flag=None, rsd=0, Nfp=None)
            _, _, _, dlogbk = Forecast.quijote_dBkdtheta(theta, log=True, z=z, dmnu=dmnu, flag=None, rsd=0, Nfp=None)
            
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
        k1, k2, k3, dbk = Forecast.quijote_dBkdtheta(theta, log=False, z=z, flag=None, rsd=0, Nfp=None)
        _, _, _, dlogbk = Forecast.quijote_dBkdtheta(theta, log=True, z=z, flag=None, rsd=0, Nfp=None)

        hdr = 'k1, k2, k3, dB/dtheta, dlogB/dtheta'
        # save to file 
        np.savetxt(fderiv, np.vstack([k1, k2, k3, dbk, dlogbk]).T, header=hdr, delimiter=',\t', fmt='%i %i %i %.5e %.5e')
    return None 


def dBdtheta_real(theta, z=0): 
    ''' write out fiducial derivatives of B with respect to theta to file in the following format: 
    k1, k2, k3, dB/dtheta, dlogB/dtheta.

    The fiducial derivatives include all B calculations: n-body, fixed-paired, different rsd 
    directions. If theta == 'Mnu', it will output all the different methods for calculating 
    derivatives: 'fin', 'fin0', 'p', 'pp', 'ppp'. 
   
    :param theta: 
        parameter value

    :param z: (default: 0) 
        redshift. currently only z=0 is implemented

    '''
    fderiv = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.%s.real.dat' % theta) 
    print("--- writing dB/d%s to %s ---" % (theta, fderiv))

    if theta == 'Mnu': 
        for i_dmnu, dmnu in enumerate(['fin', 'fin0', 'p', 'pp', 'ppp']): 
            # calculate dB/dtheta and dlogB/dtheta 
            k1, k2, k3, dbk = Forecast.quijote_dBkdtheta(theta, log=False, z=z, dmnu=dmnu, flag=None, rsd='real', Nfp=None)
            _, _, _, dlogbk = Forecast.quijote_dBkdtheta(theta, log=True, z=z, dmnu=dmnu, flag=None, rsd='real', Nfp=None)
            
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
        k1, k2, k3, dbk = Forecast.quijote_dBkdtheta(theta, log=False, z=z, flag=None, rsd='real', Nfp=None)
        _, _, _, dlogbk = Forecast.quijote_dBkdtheta(theta, log=True, z=z, flag=None, rsd='real', Nfp=None)

        hdr = 'k1, k2, k3, dB/dtheta, dlogB/dtheta'
        # save to file 
        np.savetxt(fderiv, np.vstack([k1, k2, k3, dbk, dlogbk]).T, header=hdr, delimiter=',\t', fmt='%i %i %i %.5e %.5e')
    return None 


def dBdtheta_ncv(theta, z=0, rsd=0): 
    ''' write out derivatives of B with respect to theta calculated with fixed paired sims to file in the following format: 
    k1, k2, k3, dB/dtheta, dlogB/dtheta.

    The fiducial derivatives include all B calculations: n-body, fixed-paired, different rsd 
    directions. If theta == 'Mnu', it will output all the different methods for calculating 
    derivatives: 'fin', 'fin0', 'p', 'pp', 'ppp'. 
   
    :param theta: 
        parameter value

    :param z: (default: 0) 
        redshift. currently only z=0 is implemented

    :param rsd: (default: 0) 
        int (0, 1, 2) specifies redshift space distoriton direction. 'real' means real-space
    '''
    if rsd != 'real': fderiv = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.%s.ncv.rsd%i.dat' % (theta, rsd)) 
    else: fderiv = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.%s.ncv.real.dat' % theta) 
    print("--- writing dB/d%s to %s ---" % (theta, fderiv))

    if theta == 'Mnu': 
        for i_dmnu, dmnu in enumerate(['fin', 'fin0', 'p', 'pp', 'ppp']): 
            # calculate dB/dtheta and dlogB/dtheta 
            k1, k2, k3, dbk = Forecast.quijote_dBkdtheta(theta, log=False, z=z, dmnu=dmnu, flag='ncv', rsd=rsd, Nfp=None)
            _, _, _, dlogbk = Forecast.quijote_dBkdtheta(theta, log=True, z=z, dmnu=dmnu, flag='ncv', rsd=rsd, Nfp=None)
            
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
        k1, k2, k3, dbk = Forecast.quijote_dBkdtheta(theta, log=False, z=z, flag='ncv', rsd=rsd, Nfp=None)
        _, _, _, dlogbk = Forecast.quijote_dBkdtheta(theta, log=True, z=z, flag='ncv', rsd=rsd, Nfp=None)

        hdr = 'k1, k2, k3, dB/dtheta, dlogB/dtheta'
        # save to file 
        np.savetxt(fderiv, np.vstack([k1, k2, k3, dbk, dlogbk]).T, header=hdr, delimiter=',\t', fmt='%i %i %i %.5e %.5e')
    return None 


def dBdtheta_reg(theta, z=0, rsd=0): 
    ''' write out derivatives of B with respect to theta calculated using nbody sims (not fixed-paired) 
    to file in the following format: k1, k2, k3, dB/dtheta, dlogB/dtheta.

    The fiducial derivatives include all B calculations: n-body, fixed-paired, different rsd 
    directions. If theta == 'Mnu', it will output all the different methods for calculating 
    derivatives: 'fin', 'fin0', 'p', 'pp', 'ppp'. 
   
    :param theta: 
        parameter value

    :param z: (default: 0) 
        redshift. currently only z=0 is implemented

    '''
    if rsd != 'real': fderiv = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.%s.reg.rsd%i.dat' % (theta, rsd)) 
    else: fderiv = os.path.join(UT.doc_dir(), 'dat', 'dBdtheta.%s.reg.real.dat' % theta) 
    print("--- writing dB/d%s to %s ---" % (theta, fderiv))

    if theta == 'Mnu': 
        for i_dmnu, dmnu in enumerate(['fin', 'fin0', 'p', 'pp', 'ppp']): 
            # calculate dB/dtheta and dlogB/dtheta 
            k1, k2, k3, dbk = Forecast.quijote_dBkdtheta(theta, log=False, z=z, dmnu=dmnu, flag='reg', rsd=rsd, Nfp=None)
            _, _, _, dlogbk = Forecast.quijote_dBkdtheta(theta, log=True, z=z, dmnu=dmnu, flag='reg', rsd=rsd, Nfp=None)
            
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
        k1, k2, k3, dbk = Forecast.quijote_dBkdtheta(theta, log=False, z=z, flag='reg', rsd=rsd, Nfp=None)
        _, _, _, dlogbk = Forecast.quijote_dBkdtheta(theta, log=True, z=z, flag='reg', rsd=rsd, Nfp=None)

        hdr = 'k1, k2, k3, dB/dtheta, dlogB/dtheta'
        # save to file 
        np.savetxt(fderiv, np.vstack([k1, k2, k3, dbk, dlogbk]).T, header=hdr, delimiter=',\t', fmt='%i %i %i %.5e %.5e')
    return None 


if __name__=="__main__": 
    thetas = ['Mnu', 'Om', 'Ob2', 'h', 'ns', 's8', 'Mmin']
    thetas = ['Mnu']
    for theta in thetas: 
        #dBdtheta_fiducial(theta, z=0)
        dBdtheta_real(theta, z=0)
        for rsd in [0, 'real']: 
            dBdtheta_ncv(theta, z=0, rsd=rsd)
            dBdtheta_reg(theta, z=0, rsd=rsd) 
