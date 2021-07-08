# zlax
Python interface for the [Zelus](https://github.com/INRIA/zelus) to [Python-JAX compiler](https://github.com/INRIA/zelus/tree/muf) and runtime libraries.
`zlax` supports also [ProbZelus](https://github.com/IBM/probzelus) and provides interfaces and runtime librairies to write probabilistic models and run inference.

## Install

```sh
# zlax
pip install . 

# zelus-jax
opam install . 
```

## Usage

### Compile zls to py
```sh
# Zelus
zeluc -I `zeluc -where`-jax -mufpy <name>.zls

# Zelus with probability
probzeluc -I `zeluc -where`-jax -mufpy <name>.zls
```

### Simulation
Run `zluciole`. See usage using the `-h` option.


## See also