from setuptools import setup, find_packages

setup(
    name="experimental_script",
    version="0.1.0",
    packages=find_packages(),
    package_data={'experimental_script': ['config/*', 'src/*']},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
    ],
)
