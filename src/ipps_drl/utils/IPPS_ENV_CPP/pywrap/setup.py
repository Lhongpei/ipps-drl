# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import os
project_root = os.path.dirname(os.path.abspath(__file__))
include_dirs = [
    os.path.join('..', 'include'),
    '/usr/include',
]

sources = [
    "env_wrapper.pyx",
    os.path.join('..', 'src', 'env.cpp'),
    os.path.join('..', 'src', 'io.cpp'),
    os.path.join('..', 'src', 'greedy.cpp'),
    os.path.join('..', 'src', 'state.cpp'),
    os.path.join('..', 'src', 'graph.cpp'),
    os.path.join('..', 'src', 'load_utils.cpp'),
]

ext = Extension(
    name="env_wrapper",
    sources=sources,
    include_dirs=include_dirs,
    language="c++",
    define_macros=[("PROJECT_ROOT_DIR", f'"{project_root}"')],
    extra_compile_args=["-std=c++17"]
)

setup(
    name="env_wrapper",
    ext_modules=cythonize([ext], language_level=3),
)
