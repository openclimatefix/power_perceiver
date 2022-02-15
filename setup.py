from distutils.core import setup
from setuptools import find_packages
from pathlib import Path

this_directory = Path(__file__).parent
install_requires = (this_directory / "requirements.txt").read_text().splitlines()
long_description = (this_directory / "README.md").read_text()

exec(open("power_perceiver/version.py").read())
setup(
    name="power_perceiver",
    version=__version__,
    packages=find_packages(),
    url="https://github.com/openclimatefix/power_perceiver",
    license="MIT License",
    company="Open Climate Fix Ltd",
    author="Jack Kelly",
    install_requires=install_requires,
    long_description=long_description,
    ong_description_content_type="text/markdown",
    author_email="jack@openclimatefix.org",
    description=("Machine learning experiments using the Perceiver IO"
                 " model to forecast the electricity system (starting with solar)"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)