################################################################################
# Probabilistic convolution and deconvolution for sequence of indepdent counters
################################################################################

import numpy as np
from tqdm import tqdm


class SOIC:
    
    def __init__(self, W ):
        self.W = W

    def deconv( self, Py, n_epochs = 50 ):

        self.n_y = Py.shape[0]-1
        
        if self.n_y % self.W != 0:
            raise "The number of rows minus one must be divisible by window size."

        self.n_x = int( self.n_y / self.W )
        
        self.dims = tuple( [self.n_x+1 for i in range(self.W)] )

        self.N = Py.shape[1]+self.W-1        
        self.alpha = np.zeros( [(self.n_x+1)**self.W, self.N-self.W+1] )

        # random init 
        self.Px = np.random.rand( self.n_x+1,self.N)
        self.Px = self.Px / np.sum( self.Px, axis=0)

        self.Py = np.zeros( Py.shape )

        #
        self.phi = []
        for i in range( self.N-self.W+1 ):
            phii = np.zeros( (self.n_x+1,(self.n_x+1)**self.W, self.W) )
            for w in range( self.W ):
                for z in range( (self.n_x+1)**self.W ):
                    ind = np.unravel_index( z, self.dims )
                    phii[ind[w],z,w] += Py[np.sum(ind),i]

            self.phi.append( phii )
        
        # run EM
        self.conv( )
        obj = [ np.sum( Py * np.log( self.Py ) ) ]

        for i in tqdm( range( n_epochs )):
            self.e_step()
            self.m_step()

            self.conv( )
            obj.append( np.sum( Py * np.log( self.Py ) ))

        self.obj = np.array( obj )

        return np.copy( self.Px ), np.copy( self.Py ), np.copy( self.obj )



    def conv( self, Px=None ):

        if Px is not None:
            self.N = Px.shape[1]
            self.n_x = Px.shape[0]-1
            self.n_y = self.n_x*self.W
            self.Px = np.copy( Px )
            self.dims = tuple( [self.n_x+1 for i in range(self.W)] )
            self.Py = np.zeros( [self.n_y + 1, self.N-self.W+1] )

        self.Py.fill( 0 )
        for i in range( self.N-self.W+1):
            for z in range( (self.n_x+1)**self.W ):
                ind = np.unravel_index( z, self.dims )

                tmp = 1
                for j in range( self.W ):
                    tmp = tmp*self.Px[ind[j],i+j]

                self.Py[ np.sum(ind),i ] += tmp

        if Px is not None:
            return np.copy( self.Py )

        
    def e_step( self ):

        norm_const = np.zeros( self.W*(self.n_x+1) )

        for i in range( self.N-self.W+1):
            norm_const.fill(0)
            for z in range( (self.n_x+1)**self.W ):
                ind = np.unravel_index( z, self.dims )

                tmp = 1
                for j in range( self.W ):
                    tmp = tmp*self.Px[ind[j],i+j]

                self.alpha[z,i] = tmp
                norm_const[ np.sum( ind ) ] += tmp

            for z in range( (self.n_x+1)**self.W ):
                ind = np.unravel_index( z, self.dims )
                self.alpha[z,i] = self.alpha[z,i] / norm_const[ np.sum(ind) ]
            

    def m_step( self ):
    
        self.Px.fill(0)
        for i in range( self.N-self.W+1):
            for w in range( self.W ):
                self.Px[:,i+w] += self.phi[i][:,:,w] @ self.alpha[:,i]

        self.Px = self.Px / np.sum( self.Px, axis=0 ) 

