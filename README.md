# aesthetic-predictor

[![Python](https://img.shields.io/pypi/pyversions/aesthetic-predictor.svg)](https://pypi.org/project/aesthetic-predictor/)
[![PyPI version](https://badge.fury.io/py/aesthetic-predictor.svg)](https://badge.fury.io/py/aesthetic-predictor)
[![Downloads](https://static.pepy.tech/badge/aesthetic-predictor)](https://pepy.tech/project/aesthetic-predictor)
[![License](https://img.shields.io/pypi/l/aesthetic-predictor.svg)](https://github.com/google/aesthetic_predictor/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract

A python package of aesthetic quality of pictures a.k.a `aesthic-predictor`.
See details: https://github.com/LAION-AI/aesthetic-predictor

## Installation

```python
pip install aesthetic_predictor
```

## How to use

```python
from aesthetic_predictor import predict_aesthetic
from PIL import Image
print(predict_aesthetic(Image.open("path/to/image")))
```
