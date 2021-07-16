__all__ = set(globals().keys()).union({"__all__"})

import jax.numpy as np
from jax.lax import cond
from typing import Any
from sys import stdin, stdout, stderr
from dataclasses import dataclass
from enum import Enum, auto
from jax.numpy import logical_not

from .muflib import Node, step, reset, init, register_pytree_node_dataclass, J_dataclass
from jax.tree_util import register_pytree_node_class


def _print(*args, end='', f=stdout,**kwargs):
    print(*args, **kwargs, end=end, file=f)
    return ()

print_char = _print
print_string = _print
print_int = _print
print_float = _print
print_newline = lambda _: _print('\n')
print_endline = lambda *args, **kwargs : _print(end='\n', *args, **kwargs)
prerr_char = lambda *args, **kwargs : _print(f=stderr, *args, **kwargs)
prerr_string = lambda *args, **kwargs : _print(f=stderr, *args, **kwargs)
prerr_int = lambda *args, **kwargs : _print(f=stderr, *args, **kwargs)
prerr_float = lambda *args, **kwargs : _print(f=stderr, *args, **kwargs)
prerr_endline = lambda : _print( '\n', f=stderr)
prerr_newline = lambda _, *args, **kwargs : _print(end='\n', f=stderr, *args, **kwargs)

def mod(x):
    return (lambda y: np.mod(x, y))
mod_float = mod

def abs(x):
  return np.abs(x)
abs_float = abs
float_abs = abs

def exp(x):
  return np.exp(x)

def int_of_char(x):
    return int(x)

def char_of_int(x):
    assert x >=0 and x <= 255
    return chr(x)

def string_of_bool(x):
    return str(x)

def bool_of_string(x):
    return bool(x)

def int_of_string(x):
    return int(x)

def string_of_float(x):
    return str(x)

def float_of_string(x):
    return float(x)

def float_of_int(x):
    return float(x)

def int_of_float(x):
    return int(x)

def ignore(x):
    x
    return ()

def fst(x):
    return x[0]

def snd(x):
    return x[1]

def max_float(x):
    return (lambda y : np.fmax(x, y))

def min_float(x):
    return (lambda y : np.fmin(x, y))

max_int = max_float
min_int = min_float
max = max_float
min = min_float

def _mod(x):
    return (lambda y : np.fmod(x, y))

def sqrt(x):
    return np.sqrt(x)

def log(x):
    return np.log(x)

def log10(x):
    return np.log10(x)

def cos(x):
    return np.cos(x)

def sin(x):
    return np.sin(x)

def tan(x):
    return np.tan(x)

def acos(x):
    return np.arccos(x)

def asin(x):
    return np.arcsin(x)

def atan(x):
    return np.arctan(x)

def atan2(x):
    return (lambda y : np.arctan2(x, y))

def cosh(x):
    return np.cosh(x)
    
def sinh(x):
    return np.sinh(x)
    
def tanh(x):
    return np.tanh(x)
    
def ceil(x):
    return np.ceil(x)
    
def floor(x):
    return np.floor(x)

def truncate(x):
    return int_of_float(x)

infinity = np.inf
neg_infinity = -infinity
nan = np.nan
epsilon_float = np.finfo(float).eps


@register_pytree_node_dataclass
class Fpclass(J_dataclass):
    pass
FP_normal = Fpclass(1)
FP_subnormal = Fpclass(2)
FP_zero = Fpclass(3)
FP_infinite = Fpclass(4)
FP_nan = Fpclass(5)
    
def classify_float(x):
    return cond(np.isinf(x),
                lambda _: FP_infinite,
                lambda _: cond(np.isnan(x),
                               lambda _: FP_nan,
                               lambda _: cond(np.equal(0, x),
                                              lambda _: FP_zero,
                                              lambda _: cond(np.isclose(0.0, x),
                                                             lambda _: FP_subnormal,
                                                             lambda _: FP_normal,
                                                             None),
                                              None),
                               None),
                None)


# type option, None already exists in Python
@register_pytree_node_dataclass
class Some(J_dataclass):
    x: Any

def frexp(x):
    return np.frexp(x)

def ldexp(x):
    return lambda y : np.ldexp(x, y)

def modf(x):
    return np.modf(x)

def read_line(_):
    return input()

def read_int(_):
    return int(read_line(None))

def read_float(_):
    return float(read_line(None))

def open_out(path):
    return open(path, "w+")

def open_out_bin(path):
    return open(path, "bw+")
    
def open_out_gen(mode, perm, path):
    assert False, "The \"mode\" argument must be converted from a \
\"type open_flag\" to a \"str\" that is accepted by Python open"
    assert False, "Need to implement a way to modify the file permissions, \
in case the file must be created. "
    def _f1(perm):
        def _f2(path):
            return open(path, mode=mode)
        return _f2
    return _f1

def open_in(path):
    return open(path, "r")

def open_in_bin(path):
    return open(path, "rb")

def open_in_gen(mode, perm, path):
    assert False, "The \"mode\" argument must be converted from a \
\"type open_flag\" to a \"str\" that is accepted by Python open"
    assert False, "Need to implement a way to modify the file permissions, \
in case the file must be created."
    def _f1(perm):
        def _f2(path):
            return open(path, mode=mode)
        return _f2
    return _f1

def flush(ch):
    ch.flush()
    return ()

def flush_all(_):
    assert False, "Not implemented in Python"

def output_char(ch):
    def _f1(c):
        ch.write(c)
        return ()
    return _f1

def output_string(ch, s):
    def _f1(s):
        ch.write(s)
        return ()
    return _f1

def ouput(ch):
    def _f1(buf):
        def _f2(pos):
            def _f3(length):
                if not (pos < len(buf) and (pos+length) < len(buf)):
                    raise ValueError("output : pos and length do not designate a valid substring of buf")
                ch.write(buf[pos:pos+length])
                return ()
            return _f3
        return _f2
    return _f1

def output_byte(ch):
    def _f1(byte):
        byte = bytes([byte])
        ch.write(byte)
        return ()
    return _f1

def output_binary_int(ch):
    def _f1(i):
        byte4 = i.to_bytes(4, 'big')
        ch.write(byte4)
        return ()
    return _f1

def output_value(ch):
    def _f1(x):
        ch.write("{}".format(x))
        return ()
    return _f1

def seek_out(ch):
    def _f1(pos):
        ch.seek(pos)
        return ()
    return _f1

def pos_out(ch):
    return ch.tell()

def out_channel_length(ch):
    assert False, "Not implemented in Python"

def close_out(ch):
    ch.close()
    return ()

def close_out_noerr(ch):
    try:
        close_out(ch)
    except:
        pass
    return ()

def set_binary_mode_out(ch):
    def _f1(b):
        assert False, "Not implemented in Python"
    return _f1

def input_char(ch):
    return ch.read(1)

def input_line(ch):
    return ch.readline()

def input_(ch):
    def _f1(buf):
        def _f2(pos):
            def _f3(length):
                if not (pos < len(buf) and (pos+length) < len(buf)):
                    raise ValueError("output : pos and length do not designate a valid substring of buf")
                s = ch.read(length)
                buf = buf[:pos] + s # Side effect in buf
                return len(s)
            return _f3
        return _f2
    return _f1

def really_input(ch):
    def _f1(buf):
        def _f2(pos):
            def _f3(length):
                if not (pos < len(buf) and (pos+length) < len(buf)):
                    raise ValueError("output : pos and length do not designate a valid substring of buf")
                s = ch.read(length)
                if len(s) > length:
                    raise EOFError("really_input")
                buf = buf[:pos] + s # Side effect in buf
                return len(s)
            return _f3
        return _f2
    return _f1

def input_byte(ch):
    return int(input_char(ch))

def input_binary_int(ch):
    i = int(ch.read(4))
    return i.to_bytes(4, 'big')

def input_value(ch):
    return input_char(ch)

def seek_in(ch):
    def _f1(pos):
        return seek_out(ch, pos)
    return _f1

def pos_in(ch):
    return pos_out(ch)


def in_channel_length(ch):
    assert False, "Not implemented in Python"

def close_in(ch):
    return close_out(ch)

def close_in_noerr(ch):
    return close_out_noerr(ch)

def set_binary_mode_in(ch):
    def _f1(b):
        assert False, "Not implemented in Python"
    return _f1


# ( * ) operator
def _s(x):
    return (lambda y : np.multiply(x, y))

# ( = ) operator
def _e(x):
    return (lambda y : x == y)

# ( <> ) operator
def _iu(x):
    return (lambda y : np.not_equal(x, y))

# ( < ) operator
def _i(x):
    return (lambda y : np.less(x, y))

# ( > ) operator
def _u(x):
    return (lambda y : np.greater(x, y))

# ( <= ) operator
def _ie(x):
    return (lambda y : np.less_equal(x, y))

# ( > ) operator
def _ue(x):
    return (lambda y : np.greater_equal(x, y))


# ( == ) operator
def _ee(x):
    return (lambda y : x is y)

# ( != ) operator
def _xe(x):
    return (lambda y : x is not y)

#  ( ~- ) operator
def _lm(x):
    return -x

def succ(x):
    return x+1

def pred(x):
    return x-1

# ( + ) operator
def _p(x):
    return (lambda y : np.add(x, y))

# ( - ) operator
def _m(x):
    return (lambda y : np.subtract(x, y))

# ( / ) operator
def _q(x):
    return (lambda y : np.floor_divide(x, y))

# ( ~-. ) operator
_lmo = _lm

# ( +. ) operator
_po = _p

# ( -. ) operator
_mo = _m

# ( *. ) operator
_so = _s

# ( /. ) operator
_qo =  _q

# ( ** ) operator
def _ss(x):
    return (lambda y : np.power(x, y) )

# ( ^ ) operator
def  _h(x):
    return (lambda y : x + y)

def _land(x):
    return (lambda y : np.bitwise_and(x, y))
land = _land

def _lor(x):
    return (lambda y : np.bitwise_or(x, y))
lor = _lor

def _lxor(x):
    return (lambda y : np.bitwise_xor(x, y))
lxor = _lxor

def _lnot(x):
    return (lambda y : np.bitwise_not(x, y))
lnot = _lnot

def _lsl(x):
    return (lambda y : np.left_shift(x, y))
lsl = _lsl

def _asr(x):
    return (lambda y : np.right_shift(x, y))
asr = _asr

def _lsr(x):
    def _(y):
        assert False, "Not implemented in Python"
    return _
lsr = _lsr

#  ( & ) operator
def _a(x):
    return (lambda y : np.logical_and(x, y))
#  ( && ) operator
_aa = _a

#  ( || ) operator
def _vv(x):
    return (lambda y : np.logical_or(x, y))


def compare(x):
    def _f(y):
        return cond(np.equal(x,y),
                    lambda _: 0,
                    cond(np.less(x, y),
                         lambda _: -1,
                         lambda _: 1,
                         None),
                    None)
    return _f


# TODO
# val ( on ) : zero -> bool -> zero
# val orz : zero -> zero -> zero

__all__ = list(set(globals().keys()) - __all__)

