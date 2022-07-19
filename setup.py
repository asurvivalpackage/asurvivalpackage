from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.19.5", 'pandas>=1.1.5', 'matplotlib>=3.3.4']

setup(
    name="asurvivalpackage",
    version="0.0.1",
    author="Hayley Smith",
    author_email="h.r.smith96@outlook.com",
    description="A package for survival analysis in Python.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/asurvivalpackage/asurvivalpackage",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6.4",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)

