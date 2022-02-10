import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml-lib",
    version="0.0.1",
    author="Iratxe Moya",
    author_email="iratxe.moya@gmail.com",
    description="A simple library for Machine Learning tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iratxeMoya/MLLib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 2-Clause License",
        "Operating System :: OS Independent",
    ],
)