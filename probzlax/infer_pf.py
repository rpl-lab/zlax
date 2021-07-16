import jax.numpy as np
from jax.tree_util import register_pytree_node_class
import jax.random as jrd
from jax.tree_util import tree_flatten, tree_unflatten
from jax.scipy.special import logsumexp
from jax.lax import cond
from probzlax.distribution import Support

from .infer import *

from jax import tree_map
def resample(arg):
    key, probs, particles = arg
    def choice(a):
        n = a.shape[0]
        idx = jrd.choice(key, np.arange(n), shape=(n,), p=probs)
        return a[idx]
    return tree_map(choice, particles)

### Inference
def infer(n):
    def infer_node(c):
        @register_pytree_node_class
        class Infer_pf(Infer):
            def init(self):
                return super().init(c, n)

            def step(self, state, i):
                values, probs, particles, (_, keys) = super().step(state, i, n)

                # Resampling
                key, subkey = jrd.split(state["key"])
                resampled = resample((subkey, probs, particles))

                return {
                    **state,
                    "particles" : resampled,
                    "proba" : (np.zeros(n), keys),
                    "key" : key
                }, Support(values, probs)

            def get_particles_number(self):
                return n

        return Infer_pf
    return infer_node




# Effective sample size
def ess(scores):
    norm = logsumexp(scores)
    scores2 = np.subtract(np.exp(scores), norm)
    num = np.sum(scores2) ** 2
    den = np.exp(logsumexp(np.multiply(2, scores2)))
    return num / den

### Inference
def infer_ess_resample(n):  
    def infer_treshold(threshold):
        _n_treshold = threshold * n
        def infer_node(c):
            @register_pytree_node_class
            class Infer_pf_ess(Infer):
                def init(self):
                    return super().init(c, n)
                
                def step(self, state, i):
                    values, probs, particles, (scores, keys) = super().step(state, i, n)

                    # Resampling
                    key, subkey = jrd.split(state["key"])
                    resampled = cond(ess(scores) < _n_treshold, 
                                    resample, 
                                    lambda t : t[2], 
                                    (subkey, probs, particles)
                    )

                    return {
                        **state,
                        "particles" : resampled,
                        "proba" : (np.zeros(n), keys),
                        "key" : key
                    }, np.transpose(np.vstack((values, probs)))

                def get_particles_number(self):
                    return n
                                
            return Infer_pf_ess
        return infer_node
    return infer_treshold

def identity(x) : return x
def const(x) : return identity(x)
def eval(x) : return identity(x)
def pair(x, y) : return (x, y)
def mult(x, y) : return x * y

# TODO : implement the Zelus functions bellow
"""
val add : float * float -> float
val ( +~ ) : float -> float -> float
val ( *~ ) : float  -> float -> float
val app : ('a -> 'b)  * 'a  -> 'b
val ( @@~ ) : ('a -> 'b)  -> 'a  -> 'b
val array : 'a  array -> 'a array
val lst : 'a list -> 'a list
val matrix : 'a array array -> 'a array array

val mat_add : Mat.mat  * Mat.mat  -> Mat.mat
val ( +@~ ) : Mat.mat  -> Mat.mat  -> Mat.mat
val mat_scalar_mult : float  * Mat.mat  -> Mat.mat
val ( $*~ ) : float  -> Mat.mat  -> Mat.mat
val mat_dot : Mat.mat  * Mat.mat  -> Mat.mat
val ( *@~ ) : Mat.mat  -> Mat.mat  -> Mat.mat
val vec_get : Mat.mat  * int -> float

val infer_noresample :
  int -S-> ('a ~D~> 'b) -S-> 'a -D-> 'b Distribution.t

val plan :
    int -S-> int -S->
      ('t1 ~D~> 't2) -S->
        't1 -D-> 't2

val infer_depth :
    int -S-> int -S->
      ('t1 ~D~> 't2) -S->
        't1 -D-> 't2 Distribution.t

val infer_subresample :
  int -S-> ('a ~D~> 'b) -S-> bool * 'a -D-> 'b Distribution.t


val gen : ('a ~D~> 'b) -S-> 'a -D-> 'b * float
"""