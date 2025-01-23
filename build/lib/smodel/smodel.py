import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator, griddata,interp1d
from scipy.stats import norm

class SModel:
    def __init__(self,p=0.3,interp="nearest",stat="uniform"):
        '''
        Parameters:
        p - probability of the energy cascading (0.0 - 0.5)
        interp - interpolation method according to scipy.interpolation ('nearest','linear','cubic')
        stat - statistical method to generate random seeds ('uniform' or 'normal')
        '''
        self.p = p
        self.interp = interp
        self.stat = stat
    
    def __getCoordinatesScipy(self, a):
        '''
        Takes an any-dimensional array to produce a list of elements  (N,M), where N is the number of points and M is the dimension coordinates len.
        For instance, a 3x3x3 hypercube will produce a list (27,3)

        Parameters:
        a - a generic np.array
        '''
        coordsStr = np.array(np.meshgrid(*tuple(np.array([np.linspace(0,1,f) for f in a.shape]).tolist()) ))
        coordsStr = np.array(list(zip(*[np.ravel(coordsStr[i]) for i in range(coordsStr.shape[0])])))
        return [tuple(p.tolist()) for p in coordsStr]

    def __genRandomValues(self,shape):
        '''
        Generates a random draw of elements in the specified shape

        Parameters:
        shape - tuple of dimensions
        '''
        if self.stat == "uniform":
            frag = np.random.rand(*shape)
        elif self.stat == "normal":
            dist = norm(loc=0,scale=1/np.sqrt(max(shape)))
            frag = dist.rvs(shape)
        frag = frag - np.average(frag)
        return frag

    def __genScale(self, n, scale,sDim):
        '''
        Generates the Energy at a given level.
        The energy is calculated from a random value and the cascading probability

        Parameters:
        n - the size of the 'layer' in a dimension
        scale - the scale to be generated (usually a power of 2)
        sDim - number of dimensions
        '''
        out = np.ones(n)
        frag = self.__genRandomValues(tuple(np.repeat(scale,sDim).tolist()))
        for d,nd in enumerate(frag.shape):
            seqEven = np.arange(0,nd,2)
            seqOdd = np.arange(1,nd,2)
            evensD = np.repeat(slice(None), len(frag.shape))
            oddsD =  np.repeat(slice(None), len(frag.shape))
            evensD[d] = seqEven
            oddsD[d] = seqOdd
            if np.random.rand(1)>0.5:
                frag[*tuple(evensD.tolist())] *= self.p
                frag[*tuple(oddsD.tolist())] *= (1.0-self.p)
            else:
                frag[*tuple(evensD.tolist())] *= self.p
                frag[*tuple(oddsD.tolist())] *= (1.0-self.p)
        if sDim==1:
            nni = interp1d(np.linspace(0,1,scale), frag,kind=self.interp)
            out = nni(np.linspace(0,1,n))
        else:
            ptsCoords = self.__getCoordinatesScipy(frag)
            tgtCoords = self.__getCoordinatesScipy(np.ones(np.repeat(n,repeats=sDim)))
            outRavel = griddata(ptsCoords, np.ravel(frag), tgtCoords,method=self.interp)
            out = outRavel.reshape(*tuple(np.repeat(n,repeats=sDim).tolist()))
        return out

    def __call__(self, n=2048,sDim = 1):
        '''
        Generates a time-series, image or other any-simensional structure, using:
            1. the cascading probability
            2. Statistical Model 
        Parameters:
        n - the size at every dimension
        sDIm - number of Dimensions

        Example:
        n =1024, sDim=2 -> 1024x1024 image
        n=64, sDim = 3 -> 64x64x64 hypercube
        '''
        output = np.ones(np.repeat(n,repeats=sDim))
        scales = int(np.log2(n))
        for s in range(scales,1,-1):
            output = (self.p)*output+(1-self.p)*output*self.__genScale(n,2**s,sDim)
        return output/np.std(output)
