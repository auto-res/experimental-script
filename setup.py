from setuptools import setup, find_packages

setup(
    name="experimental-script",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
)
