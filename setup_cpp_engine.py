from setuptools import setup, Extension
import pybind11
import sys
import os

extra_link_args = [] if sys.platform == "darwin" else ['-static-libstdc++', '-static-libgcc']

ext_modules = [
    Extension(
        'ebola_stochastic_ring_cpp',
        ['scripts/ebola_stochastic_ring_cpp.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++11', '-Wall', '-D_GLIBCXX_USE_CXX11_ABI=0'],
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='ebola_stochastic_ring_cpp',
    ext_modules=ext_modules,
)
