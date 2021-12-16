from typing import Union
from PIL import Image
from torchvision import transforms
from torch import Tensor

img_class = Image.Image

# from keras.preprocessing.image import img_to_array
# IMAGE_INPUT_SIZE = (64, 3, 7, 7)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

normalize_only = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def img_to_tensor(img: Union[img_class, Tensor]):
    if isinstance(img, Tensor):
        return normalize_only(img)

    elif isinstance(img, img_class):
        return preprocess(img).unsqueeze(0)

    else:
        raise ValueError("Passed object was not a PIL png file,"
                         " could not convert to array")
