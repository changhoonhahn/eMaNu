'''
'''
import numpy as np 


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
    

