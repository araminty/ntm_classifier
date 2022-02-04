import os
from ntm_classifier.image_from_document import (
    page_extract_xml_figures, extract_page_images_bboxes)
from typing import Union
import PIL
from PIL.Image import Image
# import torch

from ntm_classifier.load_resources import (
    # load_mappings,
    load_model,
    process_mappings_group,
)
from ntm_classifier.extract import extract_page_images
from ntm_classifier.preprocess import img_to_tensor
from torch import no_grad, cuda, Tensor


class lazy_model:
    # Doing some lazy loading to try to solve
    # vscode unit tests freezing up while checking
    # import validity

    def __init__(self, filename=None):
        self.model = None
        self.filename = filename
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
            self.model.to(self.device)

        if self.model is None:
            if self.filename is None:
                self.model = load_model()
            else:
                self.model = load_model(self.filename)

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


def classify_page_from_xml(
        page: Image,
        xml):
    page_bbox = xml.get('bbox')
    figure_elements = page_extract_xml_figures(xml)
    coordinates = [fig.get('bbox') for fig in figure_elements]

    bbox_image_dict = extract_page_images_bboxes(
        page, coordinates, page_bbox)

    return {k: classify(v) for (k, v) in bbox_image_dict.items()}


def classify_directory(directory_path):
    if not os.path.isdir(directory_path):
        return {'error': f"{directory_path} is not a directory"}

    file_names = os.listdir(directory_path)

    def gen():
        for file in file_names:
            if file[-3:] == 'png':
                path = os.path.join(directory_path, file)
                with PIL.Image.open(path) as img:
                    img = img.crop((0, 0, img.size[0], img.size[1]))

                yield path, classify(img)

    return {k: v for (k, v) in gen()}
