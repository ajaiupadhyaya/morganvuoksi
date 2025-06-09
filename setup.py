"""
Setup file for quant-finance package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="quant-finance",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive quantitative finance system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quant-finance",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "quant-finance=src.main:main",
            "quant-backtest=run_backtest:main",
            "quant-dashboard=src.dashboard.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "quant_finance": [
            "config/*.yaml",
            "models/*.pth",
            "data/*.csv",
        ],
    },
) 