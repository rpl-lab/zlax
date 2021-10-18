from setuptools import setup

setup(
    name="zlax",
    version="2.2",
    packages=["zlax", "probzlax"],
    scripts=['zluciole'],
    description="Librairies for the Python backend of Zelus using JAX.",
    author="Reyyan Tekin",
    url="https://github.com/rpl-lab/zlax",
    install_requires=["jax<=0.2.21", "distrax"]
)
