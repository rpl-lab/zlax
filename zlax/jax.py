from zlax.std import *
from jax import grad as jgrad
from jax import vmap as jvmap
from jax.numpy import array, empty


def grad(f):
    def _step(instance, args):
        s, o = step(instance, args)
        return o, s # reverse order
    grad_fun = jgrad(_step, argnums=1, has_aux=True)

    @register_pytree_node_class
    class Grad_node(Node):
        def init(self):
            return { "state_f" : init(f) }

        def step(self, state, i):
            o, s = grad_fun(state["state_f"], i)
            
            return { 
                **state, 
                "state_f" : s
            }, o
    return Grad_node


def vmap(f, n):
    vmap_fun = jvmap(step)
    
    @register_pytree_node_class
    class Vmap_node(Node):
        def init(self):
            return { "state_f" : jvmap(lambda _: init(f)) (empty(n)) }

        def step(self, state, i):
            s, o = vmap_fun(state["state_f"], i)
            
            return { 
                **state, 
                "state_f" : s
            }, o

    return Vmap_node


def array(x):
    return array(*x)