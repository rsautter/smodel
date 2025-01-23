from setuptools import find_packages, setup

setup(
    name='SModel',
    version='0.1',
    url='https://github.com/rsautter/smodel',
    author='Rubens Andreas Sautter',
    author_email='rubens.sautter@gmail.com',
    keywords='Extreme-Events Image Hypercube  Time-series Synthetic',
    packages=find_packages(),
    install_requires=['scipy']
)
