import glob
import os.path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

include_dirs = os.path.dirname(os.path.abspath(__file__))

source_cpp = glob.glob(os.path.join(include_dirs, '*.cpp'))

setup(
    name="beamsearchcode128",
    ext_modules=[
        CppExtension("beamsearchcode128", sources=source_cpp, include_dirs=[include_dirs])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
