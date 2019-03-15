'''
'''
import numpy as np 
# -- mpl -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def Fij(dmudts, Cinv): 
    ''' given derivative of observable along thetas and
    inverse covariance matrix, return fisher matrix Fij

    :param dmudts: 
        List of derivatives d mu/d theta_i 

    :param Cinv: 
        precision matrix 

    :return Fij: 
        Fisher matrix 
    '''
    ntheta = len(dmudts) 
    Fij = np.zeros((ntheta, ntheta))
    for i in range(ntheta): 
        for j in range(ntheta): 
            dmu_dt_i, dmu_dt_j = dmudts[i], dmudts[j]

            # calculate Mij 
            Mij = np.dot(dmu_dt_i[:,None], dmu_dt_j[None,:]) + np.dot(dmu_dt_j[:,None], dmu_dt_i[None,:])
            Fij[i,j] = 0.5 * np.trace(np.dot(Cinv, Mij))
    return Fij 
    

def plotEllipse(Finv_sub, sub, theta_fid_ij=None, color='C0'): 
    ''' Given the inverse fisher sub-matrix, calculate ellipse parameters and
    add to subplot 
    '''
    theta_fid_i, theta_fid_j = theta_fid_ij
    # get ellipse parameters 
    a = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) + np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
    b = np.sqrt(0.5*(Finv_sub[0,0] + Finv_sub[1,1]) - np.sqrt(0.25*(Finv_sub[0,0]-Finv_sub[1,1])**2 + Finv_sub[0,1]**2))
    theta = 0.5 * np.arctan2(2.0 * Finv_sub[0,1], (Finv_sub[0,0] - Finv_sub[1,1]))
    for ii, alpha in enumerate([2.48, 1.52]):
        e = Ellipse(xy=(theta_fid_i, theta_fid_j), 
                width=alpha * a, height=alpha * b, angle=theta * 360./(2.*np.pi))
        sub.add_artist(e)
        if ii == 0: alpha = 0.7
        if ii == 1: alpha = 1.
        e.set_alpha(alpha)
        e.set_facecolor(color) 
    return sub
