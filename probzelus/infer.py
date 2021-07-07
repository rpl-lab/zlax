from abc import ABC, abstractmethod
import jax.numpy as np
from jax.tree_util import register_pytree_node_class
from jax import vmap
import jax.random as jrd
from jax.scipy.special import logsumexp

from zlax.muflib import Node, init, step


def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    return helper

# Initial pseudo-random key
@call_counter
def initial_key():
    return jrd.PRNGKey(initial_key.calls)

### Sample
class Sample(Node):
    def init(self):
        return {}
    def step(self, _, i):
        prob, d = i
        return self.sample(prob, d)
    
    @classmethod
    def sample(self, prob, d):
        score, key = prob
        key, subkey = jrd.split(key)
        return ((score, key), d.sample(seed=subkey))
sample = Sample.sample

### Observe
class Observe(Node):
    def init(self):
        return {}
    def step(self, _, i):
        prob, d, x = i
        return self.observe(prob, d, x)
    
    @classmethod
    def observe(self, prob, d, x):
        score, key = prob
        score += d.log_prob(x)
        return (score, key), ()
observe = Observe.observe

### Factor
class Factor(Node):
    def init(self):
        return {}
    def step(self, _, i):
        prob, f0 = i
        return self.factor(prob, f0)
    
    @classmethod
    def factor(self, prob, f0):
        (score, key) = prob
        return (score + f0, key), ()
factor = Factor.factor


# Get the number of vectorized instances
def get_vectorization_size(node):
    l = []
    for k,v in node.state.items():
        try :
             l.append(v.get_particles_number())
        except AttributeError:
            pass
    if l == [] :
        raise AttributeError("Node {} has no vectorized nodes".format(node))
    elif all(elem == l[0] for elem in l):
        return l[0]
    else:
        raise ValueError("Node {} has mulitple vectorized nodes with different number of nodes. \
Maybe the `get_vectorization_table()` method is more appropriate for your case.")

def get_vectorization_table(node):
    d = dict()
    for k,v in node.state.items():
        try :
            d[v] = v.get_particles_number()
        except AttributeError:
            pass
    return d

### Infer
@register_pytree_node_class
class Infer(Node, ABC):

    @abstractmethod
    def init(self, c, n):
        key, *subkeys = jrd.split(initial_key(), num=n+1)
        proba = np.zeros(n), np.array(subkeys)
        vector_c_init = vmap(lambda _: init(c))(np.empty(n))
        
        return {
            "proba" : proba,
            "particles" : vector_c_init,
            "key" : key
        }

    @abstractmethod
    def step(self, state, i, n):
        vector_i = vmap(lambda _ : i)(np.empty(n))

        tab = vmap(step)(state["particles"], (state["proba"], vector_i))
        particles, ((scores, keys), values) = tab

        probs = np.exp(scores - logsumexp(scores)) 

        return values, probs, particles, (scores, keys)

    @abstractmethod
    def get_particles_number(self):
        raise NotImplementedError ("Abstract method get_particles_number not implemented.")
    
