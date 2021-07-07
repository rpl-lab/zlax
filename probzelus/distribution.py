from distrax import Normal, Uniform, Bernoulli
from jax.random import PRNGKey
from jax.random import split as jrd_split
from jax.random import beta as jrd_beta
from jax.random import poisson as jrd_poisson
from jax.scipy.special import gammaln
import jax.numpy as np

def gaussian (mean, sigma2): return Normal(loc=mean, scale=np.sqrt(sigma2))
def uniform_float (low, high): return Uniform(low=low, high=high)
def bernoulli (p): return Bernoulli(probs=p)


class Beta():
    def __init__(self, a, b):
        self.a, self.b = a, b

    def sample(self, seed):
        return jrd_beta(seed, self.a, self.b)

    def log_prob(self, value):
        assert False

    def mean(self):
        return self.a / (self.a + self.b)

    def variance(self):
        total = self.a + self.b
        return self.a * self.b / (total ** 2 * (total + 1))
def beta (a, b) : return Beta(a, b)


class Poisson():
    def __init__(self, lambd):
        self.lambd = lambd

    def sample(self, seed):
        return jrd_poisson(seed, self.lambd)

    def log_prob(self, value):
        return (np.log(self.lambd) * value) - gammaln(value + 1) - self.lambd

    def mean(self):
        return self.lambd

    def variance(self):
        return self.lambd
def poisson (lambd) : return Poisson(lambd)


def score(distrib, x):
    return distrib.log_prob(x)

def seed(seed=0):
    def decorator(func):
        def aux(*args, **kwargs):
            aux._k, aux.key = jrd_split(aux._k)
            return func(*args, **kwargs)
        aux._k, aux.key = jrd_split(PRNGKey(seed))
        return aux
    return decorator

@seed()
def draw(distrib, key=None):
    """
    If draw is vectorized, draw need an argument key in order to be able 
    to draw different values in parallel. If this argument is not given, 
    the same key may be used for each instance of the vectorized draw.
    """
    def _draw(d, k):
        if k is None:
            k = draw.key
        return d.sample(seed=k)
    return _draw(distrib, key)


def mean_float(t):
    return np.average(t[:,0], weights=t[:,1])

def stats_float(t):
    mean = mean_float(t)
    mu = np.average(np.square(t - mean))
    return mean, mu

def split(x):
    p = x[:,2]
    return np.column_stack((x[:,0], p)), np.column_stack((x[:,1], p))

# TODO : implement the Zelus functions bellow
"""
open Zelus_owl

(* type proba = float *)
type log_proba = float
type 'a t (* = *)
  (*   Dist_sampler of ((unit -> 'a) * ('a -> float)) *)
  (* | Dist_sampler_float of *)
  (*     ((unit -> float) * (float -> float) * (unit -> float * float)) *)
  (* | Dist_support of ('a * float) list *)
  (* | Dist_mixture of ('a t * float) list *)
  (* (\* | Dist_pair of 'a t * 'b t -> ('a * 'b) t *\) *)
  (* | Dist_list of 'a t list *)
  (* | Dist_array of 'a t array *)
  (* | Dist_gaussian of float * float *)
  (* | Dist_beta of float * float *)
  (* | Dist_bernoulli of float *)
  (* | Dist_uniform_int of int * int *)
  (* | Dist_uniform_float of float * float *)
  (* | Dist_exponential of float *)
  (* | Dist_add of float t * float t *)
  (* | Dist_mult of float t * float t *)
  (* | Dist_app : ('a -> 'b) t * 'a t -> 'b t *)

val draw : 'a t -> 'a
val score : 'a t * 'a -> float
val draw_and_score : 'a t -> 'a * float

val print_any_t : 'a t -AD-> unit
val print_float_t : float t -AD-> unit
val print_int_t : int t -AD-> unit
val print_bool_t : bool t -AD-> unit
val print_t : ('a -> string) -S-> 'a t -AD-> unit

val stats_float_list : float list t -> (float * float) list
val mean_float_list : float list t -> float list
val mean : ('a -> float) -> 'a t -> float
val mean_list : ('a -> float) -> 'a list t -> float list
val mean_int : int t -> float
val mean_bool : bool t -> float
val mean_signal_present : 'a signal t -> float
val mean_matrix: Mat.mat t -> Mat.mat

val dirac : 'a -> 'a t

val bernoulli_draw : float -> bool
val bernoulli_score : float -> bool -> float
val bernoulli_mean : 'a -> 'a
val bernoulli_variance : float -> float
val bernoulli : float -> bool t

val gaussian_draw : float -> float -> float
val gaussian_score : float -> float -> float -> float
val gaussian_mean : 'a -> 'b -> 'a
val gaussian_variance : 'a -> float -> float
val gaussian : float * float -> float t

val mv_gaussian : Mat.mat * Mat.mat -> Mat.mat t
val mv_gaussian_curried : Mat.mat -S-> Mat.mat -> Mat.mat t

val sph_gaussian : float list * float list -> float list t

val uniform_int_draw : int -> int -> int
val uniform_int_score : int -> int -> int -> float
val uniform_int_mean : int -> int -> float
val uniform_int_variance : int -> int -> float
val uniform_int : int * int -> int t

val uniform_float_draw : float -> float -> float
val uniform_float_score : float -> float -> float -> float
val uniform_float_mean : float -> float -> float
val uniform_float_variance : float -> float -> float
val uniform_float : float * float -> float t

val uniform_list : 'a list -> 'a t

val weighted_list : (float * 'a) list -> 'a t

val shuffle : 'a list -> 'a list t

val exponential_draw : float -> float
val exponential_score : float -> float -> float
val exponential_mean : float -> float
val exponential_variance : float -> float
val exponential : float -> float t

val poisson_draw : float -> int
val poisson_score : float -> float -> float
val poisson_mean : float -> float
val poisson_variance : float -> float
val poisson : float -> int t

val alias_method_unsafe : 'a array -> float array -> 'a t
val alias_method_list : ('a * float) list -> 'a t
val alias_method : 'a array -> float array -> 'a t

val add : float t * float t -> float t
val mult : float t * float t -> float t
val app : ('a -> 'b) t * 'a t -> 'b t

val to_dist_support : 'a t -> 'a t

val of_list : 'a t list -> 'a list t
val of_pair : 'a t * 'b t -> ('a * 'b) t
val split : ('a * 'b) t -> 'a t * 'b t
val split_array : 'a array t -> 'a t array
val split_matrix : 'a array array t -> 'a t array array
val split_list : 'a list t -> 'a t list
val to_mixture : 'a t t -> 'a t
val to_signal : 'a signal t -> 'a t signal
val map : ('a -> 'b) -> 'a t -> 'b t
"""