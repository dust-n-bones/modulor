from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
import numpy
#
# setup(
#     ext_modules=[
#         Extension("my_module", ["my_module.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )

setup(
    ext_modules=cythonize("maco_implementations/maco_full_parallel.pyx"),
    include_dirs=[numpy.get_include()], requires=['SharedArray']
)