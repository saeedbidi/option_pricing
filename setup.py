import os

from setuptools import find_namespace_packages, setup


setup(
    name="derivative-pricing",
    version='1.0',
    packages=find_namespace_packages(),
    include_package_data=True,
    url="https://github.com/saeedbidi/option_pricing",
    description="pricing European call and put options using various models",
    author="Saeed Bidi",
    author_email="saeed.bidi@qmul.ac.uk",
    python_requires=">=3.8",
    install_requires=[
        "numpy==2.1.1",
        "yfinance==0.2.43",
        "scipy==1.14.1",
        "matplotlib==3.9.2",
        "streamlit==1.38.0",
        "seaborn==0.13.2",
        "plotly==5.24.1",
        "scikit-learn==1.5.2"
    ],)
