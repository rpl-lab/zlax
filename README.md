# zlax

To install the `zlax` Python package, run `pip install .`.

To install the `zelus-jax` Zelus package, run `opam install .`

zls to py compilation : 
```
# Zelus
zeluc -I `zeluc -where`-jax -mufpy <name>.zls

# Zelus with probability
probzeluc -I `zeluc -where`-jax -mufpy <name>.zls
```

Simulate zelus code : `zluciole`. See usage using the `-h` option.
