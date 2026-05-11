# setup.py
"""
Setup script for SDF Microcirculation Analysis package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sdf-microcirculation",
    version="1.0.0",
    author="Boyuan Peng",
    author_email="burrypeng@gmail.com",
    description="Automated analysis pipeline for SDF microcirculation videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/SDFcode",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.5.0",
        "opencv-contrib-python>=4.5.0",
        "numpy>=1.20.0",
        "scikit-image>=0.18.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "Pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
    },
)
