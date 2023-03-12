from functools import lru_cache
from pathlib import Path
from typing import Any, List, Literal, Tuple, Union
from urllib.request import urlretrieve

import open_clip
import torch
import torch.nn as nn
from PIL.Image import Image

ModelType = Literal["large", "small"]
_model_name_to_clip_model = {"large": "ViT-L-14", "small": "Vit-B-32"}
_model_name_to_aesthetic_model = {
    "large": "vit_l_14",
    "small": "vit_b_32",
}


@lru_cache(1)
def get_aesthetic_model(model_type: ModelType = "large") -> nn.Module:
    """load the aethetic model"""
    cache_folder = Path.home() / ".cache/aesthetic_predictor"
    aesthetic_model_type = _model_name_to_aesthetic_model[model_type]
    path_to_model = cache_folder / f"sa_0_4_{aesthetic_model_type}_linear.pth"

    if model_type == "large":
        m = nn.Linear(768, 1)
    elif model_type == "small":
        m = nn.Linear(512, 1)
    else:
        raise KeyError(f"Not implemented model name: {model_type}")

    if not path_to_model.exists():
        cache_folder.mkdir(exist_ok=True, parents=True)
        url_model = f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_{aesthetic_model_type}_linear.pth?raw=true"
        urlretrieve(url_model, path_to_model)
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m


@lru_cache(1)
def get_clip_model(model_type: ModelType = "large") -> Tuple[open_clip.CLIP, Any]:
    """load the clip model"""

    model, _, preprocessor = open_clip.create_model_and_transforms(
        _model_name_to_clip_model[model_type], pretrained="openai"
    )
    return model, preprocessor


def predict_aesthetic(
    image: Union[Image, List[Image]],
    model_type: ModelType = "large",
) -> torch.Tensor:
    """Predict the aesthetic score of an image

    Parameters
    ----------
    image : Union[Image, List[Image]]
        input image
    model_type : ModelType, optional
        model type, by default "large"

    Returns
    -------
    torch.Tensor
        Aesthic score
    """
    aesthetic_model = get_aesthetic_model(model_type)
    clip, preprocessor = get_clip_model(model_type)

    if isinstance(image, list):
        image = torch.stack([preprocessor(img) for img in image], dim=0)
    else:
        image = preprocessor(image)[None]

    with torch.no_grad():
        image_features = clip.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = aesthetic_model(image_features)
    return prediction
