from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
        name="cython_score_evaluate",
        ext_modules=cythonize("cython_score_evaluate.pyx"),
        zip_safe=False,
        include_dirs=[np.get_include()]
        )
