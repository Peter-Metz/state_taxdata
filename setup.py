import setuptools
import os

# with open("README.md", "r") as f:
#     long_description = f.read()

setuptools.setup(
    name="dataprep",
    version=os.environ.get("VERSION", "0.0.0"),
    author="Peter Metz",
    author_email="pmetzdc@gmail.com",
    description=(
        "Prepares tax data for nonlinear optimization."
    ),
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/Peter-Metz/state_taxdata",
    packages=setuptools.find_packages(),
    install_requires=["taxcalc", "paramtools"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)