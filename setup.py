"""
Created on January 11, 2024

@author: Paula G
"""

import setuptools

with open('requirements.txt') as requirements:
    required = requirements.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AdDownloader", 
    version="0.1.0",
    author="Paula-Alexandra Gitu",
    author_email="paularossi1000@gmail.com",
    description="A cmd tool for downloading ads and their media content.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires = required,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: UM License",
        "Operating System :: Microsoft :: Windows",
        'Development Status :: 3 - Alpha'
    ],
    python_requires='==3.9.*',
    entry_points={
        "console_scripts": [
            "AdDown = AdDownloader.cli:main"
        ]
    }
)