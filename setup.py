from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="concatcpp",
      ext_modules=[cpp_extension.CppExtension('concatcpp', ['concat.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
