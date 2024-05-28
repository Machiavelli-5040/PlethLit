import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = Extension(
    "base_dtw",
    sources=["base_dtw.pyx"],
    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API",)],
)

setup(name="base_dtw", ext_modules=cythonize(extensions))
