"""Setup configuration for neuromorphic-sda package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neuromorphic-sda",
    version="0.1.0",
    author="Yash Verma",
    author_email="yashverma25104@gmail.com",
    description=(
        "A research pipeline for deep-space satellite detection using "
        "neuromorphic event cameras and Spiking Neural Networks"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YashVermaTech/neuromorphic-sda",
    project_urls={
        "Bug Tracker": "https://github.com/YashVermaTech/neuromorphic-sda/issues",
        "Documentation": "https://github.com/YashVermaTech/neuromorphic-sda/tree/main/docs",
        "Portfolio": "https://yashverma-ai.netlify.app",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "nbconvert>=7.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nsda-pipeline=data_pipeline.orbital_to_events:main",
            "nsda-benchmark=benchmarks.deterministic_env:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
