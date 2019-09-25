from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    # ext_modules = [Extension("gibbs_sampling", ["gibbs_sampling.pyx"], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])],
    ext_modules = [Extension("gibbs_sampling", ["gibbs_sampling.pyx"], extra_compile_args=['-Xpreprocessor', '-fopenmp'], extra_link_args=['-lomp'])],
    include_dirs= [np.get_include()]
)
