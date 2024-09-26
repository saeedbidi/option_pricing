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
    install_requires=["numpy", "yfinance", "scipy", "matplotlib", "streamlit", "seaborn", "plotly"],
)
