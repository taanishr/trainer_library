from jax import numpy as jnp
from jax import vmap
import numpy as np

class optim:
    @staticmethod
    def sqrtm(C):
        # Computing diagonalization
        evalues, evectors = jnp.linalg.eig(C)
        # Ensuring square root matrix exists
        sqrt_matrix = evectors @ jnp.diag(jnp.sqrt(evalues)) @ jnp.linalg.inv(evectors)
        return sqrt_matrix.real
    
    @staticmethod
    def nonzero_sorted_eig(A, eps=1e-5):
        w,v = jnp.linalg.eig(A)
        w = w.at[w < eps].set(0)
        s_ind = jnp.argsort(w)
        idx = s_ind[jnp.in1d(s_ind, jnp.flatnonzero(w!=0))]
        return w[idx].real, v[:,idx].real

    @staticmethod
    def stproject(x):
        U_x, _, V_xt = jnp.linalg.svd(x, full_matrices=False)
        return U_x@V_xt
    
    @staticmethod
    def F(X, A, B, C):
        """Lagrangian - 116 """
        C_sqrt = optim.sqrtm(C)
        return  jnp.trace(jnp.dot(X.T, A(X)@C)) + 2*jnp.trace(jnp.dot(X.T, B@C_sqrt))

    @staticmethod
    def value_and_gradF(X, A, B, C):
        """grad F """
        C_sqrt = optim.sqrtm(C).real
        return optim.F(X, A, B, C), A(X)@C + B@C_sqrt

    @staticmethod
    def foc(X, A, B, C):
        C_sqrt = optim.sqrtm(C).real
        L = X.T@(A(X)@C + B@C_sqrt)
        return np.linalg.norm(A(X)@C + B@C_sqrt - X@L)**2, L
    
    @staticmethod
    def line_search(s, Gk, Xk, L_, B_, C):
        Xk = optim.stproject(Xk - s*Gk)
        _f_k = optim.F(Xk, L_, B_, C)
        return _f_k, Xk
    
    @staticmethod
    def optim(X, A, B, C, step_sizes):
        Xk = np.array(X)
        FKs = []
        FOCs = []
        lmaxs = []
        for k in range(21):
            Fk, Gk = optim.value_and_gradF(Xk, A, B, C)
            _Fks, _Xks = vmap(optim.line_search,in_axes=(0,None,None,None,None,None))(step_sizes,Gk,Xk,A,B,C)
            _f_kamin = np.argmin(_Fks)
            _f_k = _Fks[_f_kamin]
            Xk = _Xks[_f_kamin]
            firstordercondition, lamb = optim.foc(Xk, A, B, C)
            Fk = optim.F(Xk, A, B, C)
            FKs.append(Fk)
            FOCs.append(firstordercondition)
            lmaxs.append(np.linalg.eigvals(lamb).max())
        
        return Xk, FKs, FOCs, lmaxs
