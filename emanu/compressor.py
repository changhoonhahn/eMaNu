'''
'''
import numpy as np 
from sklearn.decomposition import PCA


class Compressor(object): 
    '''
    '''
    def __init__(self, method=None): 
        '''
        '''
        if method not in ['PCA', 'KL']: 
            raise NotImplementedError
        self.method = method 
        self.Xbar = None # average data vector
        self.Xscale = None # component wise variance
        self.cM = None # compression matrix

    def fit(self, X, *args, **kwargs):  
        ''' fit the compression 
        
        Parameters 
        ----------
        X : array 
            (N x Ndim) array to fit the compression 
        mean_sub : bool
            If True, subtract mean from data vector before fitting compression  
        '''
        if self.method == 'KL': 
            self._KL_fit(X, *args, **kwargs) 
        elif self.method == 'PCA': 
            self._PCA_fit(X, *args, **kwargs) 
        else: 
            raise NotImplementedError
        return None 

    def transform(self, Y): 
        ''' transform (compress) input array using the compression
        matrix calculated in `self.fit`

        Parameters
        ----------
        Y : array
            (N x Ndim) array that you want compressed. 

        Returns 
        -------
        cY : array 
            (N x Ncomp) compressed array 
        '''
        if self.method == 'PCA': 
            #cY = self._pca.transform(Y)
            cY = np.dot(Y, self.cM.T) 
        else: 
            if self.cM is None: raise ValueError('first fit the compression') 
            assert self.cM.shape[1] == Y.shape[1] 
        
            cY = np.dot(self.cM, Y.T).T 
        return cY           

    def _KL_fit(self, X, dxdt):
        ''' fit KL/MOPED compression matrix. For our purposes of Gaussian 
        likelihood, this is the same as score compression 
    
        Parameters 
        ----------
        X : array 
            (N, Ndim) numpy array of data vector. These will be used to 
            calculate the covariance and precision matricies 
        dxdt : array 
            (Ntheta x Ndim) numpy array of the derivative w.r.t. theta
            d<x>/dtheta. 
        
        Returns
        -------
        cX : array 
            (N x Ntheta) numpy array of compressed data vector
        dcXdt : array
            (Ntheta x Ntheta) numpy array of the compressed derivative 

        References
        ----------
        * Tegmark et al. (1997) 
        * Heavens et al. (2000) 
        * Gualdi et al. (2018) 
        * Alsing et al. (2019) 
        '''
        assert X.shape[1] == dxdt.shape[1]
        ntheta = dxdt.shape[0] # number of parameters 

        Cx = np.cov(X.T) 
        if np.linalg.cond(Cx) >= 1e16: print('Covariance matrix is ill-conditioned') 
        iCx = np.linalg.inv(Cx) 

        ndata       = X.shape[1]
        nmock       = X.shape[0]
        f_hartlap   = float(nmock - ndata - 2)/float(nmock - 1) 
        iCx         *= f_hartlap
        
        B = np.zeros((ntheta, X.shape[1]))  
        for itheta in range(ntheta): 
            dxdt_i = dxdt[itheta]
            B_i = np.dot(iCx, dxdt_i)  
            B[itheta,:] = B_i
        self.cM = B.copy() 
        return None 

    def _PCA_fit(self, X, n_components=6, whiten=False): 
        ''' fit PCA compression matrix. 

        Parameters
        ----------
        X : array 
            (N, Ndim) numpy array of data vector. 
        n_components : int 
            (default: 6) number of PCA components 
        '''
        #self.Xbar = np.mean(X, axis=0) 
        #X -= self.Xbar
        Cx = np.cov(X.T)  
        l, principal_axes = np.linalg.eig(Cx) 
        isort = l.argsort()[::-1]
        self.cM = principal_axes[:n_components,isort]
        #self._pca = PCA(copy=True, n_components=n_components, whiten=whiten)
        #self._pca.fit(X) 
        #self.cM = self._pca.components_
        return None 
