import os

from setuptools import find_namespace_packages, setup


setup(
    name="derivative-pricing",
    version='0.1',
    packages=find_namespace_packages(),
    include_package_data=True,
    # url="https://github.com/",
    description="pricing European call and put options using the Black-Scholes model",
    author="Saeed Bidi",
    author_email="saeed.bidi@qmul.ac.uk",
    python_requires=">=3.8",
    install_requires=["numpy", "yfinance", "scipy", "matplotlib"],
)
