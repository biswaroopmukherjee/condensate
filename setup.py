import setuptools
from skbuild import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


with open('LICENSE') as f:
    license = f.read()

setup(
    name="condensate", # Replace with your own username
    version="0.0.1",
    author="Biswaroop Mukherjee",
    author_email="mail.biswaroop@gmail.com",
    description="Python-wrapped C++/CUDA accelerated numerical solutions of the GP equation ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/biswaroopmukherjee/condensate",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.6',
    cmake_source_dir='condensate/core'
)