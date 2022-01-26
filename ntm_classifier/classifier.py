from typing import Union
from PIL.Image import Image
# import torch

from ntm_classifier.load_resources import (
    # load_mappings,
    load_primary,
    process_mappings_group,
)
from ntm_classifier.extract import extract_page_images
from ntm_classifier.preprocess import img_to_tensor
from torch import no_grad, cuda, Tensor


class lazy_model:
    # Doing some lazy loading to try to solve
    # vscode unit tests freezing up while checking
    # import validity

    def __init__(self):
        self.model = None
        self.device = 'cuda' if cuda.is_available() else 'cpu'

    def classify(self, input_tensor, model=None):
        if not check_invalid_shape(input_tensor):
            if (len(input_tensor) == 3 and
                    check_invalid_shape(input_tensor.unsqueeze(0))):
                return self.classify(input_tensor.unsqueeze(0), model)
            else:
                raise ValueError(
                    "Invalid image tensor shape passed to "
                    "classifier.  Expected 4 dimensions got shape "
                    f"of {input_tensor.shape}.")

        if model is not None:
            self.model = model

        if self.model is None:
            self.model = load_primary()

        with no_grad():
            result = self.model(input_tensor.to(self.device))

        return result.argmax(1).item()


model_wrapper = lazy_model()
primary_mappings = process_mappings_group('primary')


def check_invalid_shape(input_tensor, max_batch_size=64):
    if not isinstance(input_tensor, Tensor):
        raise ValueError("Classifier function was "
                         "passed a non tensor object")
    shape = input_tensor.shape
    # conforms to RGB input channels
    three_channels = (len(shape) > 0) and (shape[1] == 3)
    # dimensions should be image, channels, x, y
    # or is it image, channels, y, x, either way, it's 4
    four_dimensions = (len(shape) == 4)

    return three_channels and four_dimensions


def classify_to_num(input_tensor, model=None):
    # results = model(input_tensor).argmax(1).item()
    # return results
    return model_wrapper.classify(input_tensor, model)


def label(value: Union[int, float]):
    return primary_mappings.get(int(value), 'Mapping not found')


def classify(img, model=None):
    input_tensor = img_to_tensor(img)
    classification = classify_to_num(input_tensor, model)
    return label(classification)


def classify_extractions_dictionary(images: dict):
    return {coordinates: classify(png)
            for (coordinates, png) in images.items()}


def classify_page(
        page: Image,
        coordinates: Union[list, tuple]):

    images = extract_page_images(page, coordinates)
    return classify_extractions_dictionary(images)
