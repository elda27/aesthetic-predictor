import numpy as np
from PIL import Image

import aesthetic_predictor


def test_get_aesthetic_model():
    model = aesthetic_predictor.get_aesthetic_model()
    assert model


def test_predict_single_image():
    image = Image.open("tests/test_image.jpg")
    score = aesthetic_predictor.predict_aesthetic(image)
    assert np.all((score.numpy() > 6.0).ravel())


def test_predict_multiple_images():
    image = Image.open("tests/test_image.jpg")
    score = aesthetic_predictor.predict_aesthetic([image for i in range(4)])
    assert np.all((score.numpy() > 6.0).ravel())
