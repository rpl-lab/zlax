def self_init(*args):
    return None


# TODO : implement the Zelus functions bellow
"""
val initialise : int -AD-> unit
(** Initialise the generator, using the argument as a seed.
     The same seed will always yield the same sequence of numbers. *)

val full_init : int array -AD-> unit
(** Same as {!Random.init} but takes more data as seed. *)

val self_init : unit -AD-> unit
(** Initialize the generator with a more-or-less random seed chosen
   in a system-dependent way. *)

val bits : unit -> int
(** Return 30 random bits in a nonnegative integer. *)

val int : int -> int
(** [Random.int bound] returns a random integer between 0 (inclusive)
     and [bound] (exclusive).  [bound] must be more than 0 and less
     than 2{^30}. *)

val int32 : Int32.t -> Int32.t;;
(** [Random.int32 bound] returns a random integer between 0 (inclusive)
     and [bound] (exclusive).  [bound] must be greater than 0. *)

val nativeint : Nativeint.t -> Nativeint.t;;
(** [Random.nativeint bound] returns a random integer between 0 (inclusive)
     and [bound] (exclusive).  [bound] must be greater than 0. *)

val int64 : Int64.t -> Int64.t;;
(** [Random.int64 bound] returns a random integer between 0 (inclusive)
     and [bound] (exclusive).  [bound] must be greater than 0. *)

val float : float -> float
(** [Random.float bound] returns a random floating-point number
   between 0 (inclusive) and [bound] (exclusive).  If [bound] is
   negative, the result is negative or zero.  If [bound] is 0,
   the result is 0. *)

val bool : unit -> bool
(** [Random.bool ()] returns [true] or [false] with probability 0.5 each. *)
"""
