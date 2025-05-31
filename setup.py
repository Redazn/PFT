from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="docpft",
    version="0.1.0",
    description="Quantum-Hybrid Document Processing with PFT Methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Muhammad Dimas Prabowo",
    author_email="your.email@example.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: General",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="quantum, document, processing, pft, hybrid, ai",
    packages=find_packages(include=["DocPFT", "DocPFT.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pennylane>=0.22.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
)
