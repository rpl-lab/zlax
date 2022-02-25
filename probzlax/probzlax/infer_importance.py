from jax.tree_util import register_pytree_node_class
from probzlax.distribution import Support

from .infer import *


### Inference
def infer(n):
    def infer_node(c):
        @register_pytree_node_class
        class Infer_imp(Infer):
            def init(self):
                return super().init(c, n)

            def step(self, state, i):
                values, probs, particles, proba = super().step(state, i, n)

                return {**state, "particles": particles, "proba": proba}, Support(
                    values, probs
                )

            def get_particles_number(self):
                return n

        return Infer_imp

    return infer_node


# TODO : implement the Zelus functions bellow
"""
val infer_decay :
  int -S-> float -S-> ('a ~D~> 'b) -S-> 'a -D-> 'b Distribution.t
"""
