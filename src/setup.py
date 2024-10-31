from setuptools import setup, Extension
import pybind11

# 定义扩展模块
example_extension = Extension(
    'simple_ml_ext',
    sources=['simple_ml_ext.cpp'],
    include_dirs=[pybind11.get_include(),],
)

# 设置setup
setup(
    name='simple_ml_ext',
    version='1.0',
    description='Pybind11 example',
    ext_modules=[example_extension]
)