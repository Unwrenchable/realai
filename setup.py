"""Setup configuration for RealAI."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="realai",
    version="1.0.0",
    author="RealAI Team",
    description="The AI model that can do it all - a comprehensive AI solution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Unwrenchable/realai",
    py_modules=["realai"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "realai=realai:main",
        ],
    },
)
