from setuptools import setup, find_packages

setup(
    name="experimental-script",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.8",
)
