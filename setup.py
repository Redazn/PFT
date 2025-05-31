from setuptools import setup, find_packages
import pathlib

# Read the long description from README.md
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="PFT",
    version="0.1.0",
    description="Quantum-Hybrid Document Processing Framework with PFT Methods",
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
    keywords="quantum, document processing, nlp, hybrid computing, pft",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8, <3.11",
    install_requires=[
        "numpy>=1.20.0",
        "pennylane>=0.22.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "mypy>=0.910",
            "flake8>=3.9",
            "black>=21.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15",
        ],
        "quantum": [
            "pennylane-sf>=0.22.0",  # Strawberry Fields plugin
            "pennylane-qiskit>=0.22.0",  # IBM Qiskit plugin
        ],
    },
    package_data={
        "qhdoc": ["py.typed"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/quantum-hybrid-doc-framework/issues",
        "Source": "https://github.com/yourusername/quantum-hybrid-doc-framework",
    },
    entry_points={
        "console_scripts": [
            "qhdoc-analytics=qhdoc.cli:analyze_documents",
            "qhdoc-train=qhdoc.cli:train_quantum_model",
        ],
    },
)
