"""
Created on January 11, 2024

@author: Paula G
"""

from setuptools import setup, find_packages

with open('requirements.txt') as requirements:
    required = requirements.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="AdDownloader", 
    version="0.2.0",
    author="Paula-Alexandra Gitu",
    author_email="paularossi1000@gmail.com",
    description="A cmd tool for downloading ads and their media content.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires = required,
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
    ],
    python_requires='==3.9.*',
    entry_points={
        "console_scripts": [
            "AdDown = AdDownloader.cli:main"
        ]
    }
)