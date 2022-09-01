# DurableConsumerType REMARK

This REMARK replicates Druedahl, J. (2021). A guide on solving non-convex consumption-saving models. Computational Economics, 58(3), 747-775.

## To Run all codes

First, create a new environment and install all required packages in requirements.txt. Then go to 
```
code/python
```

## To Reproduce all files

After installing the requirements described above, all results of the project should be reproducible by executing the following command from the command line in this directory
```
$ nbreproduce reproduce.sh
```
or similarly for any other available script (e.g., `nbreproduce reproduce_min.sh`)

### How to install nbreproduce?

Detailed documentation about `nbreproduce` is at [https://econ-ark.github.io/nbreproduce/](https://econ-ark.github.io/nbreproduce/).

`nbreproduce` is available through PyPI and depends on [Docker](https://www.docker.com/products/docker-desktop).

If you already have Docker installed locally you can install `nbreproduce` using
```
$ pip install nbreproduce # make sure you have Docker installed on your machine.

