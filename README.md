# aesthetic-predictor

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
