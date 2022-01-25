from typing import Union
from PIL.Image import Image
# import torch

from ntm_classifier.load_resources import load_mappings, load_primary
from ntm_classifier.extract import extract_page_images
from ntm_classifier.preprocess import img_to_tensor


model = load_primary()
mappings = load_mappings()
primary_mappings = mappings.get('primary', {})
primary_mappings = {int(k): v for (k, v) in primary_mappings.items()}


def classify_to_num(input_tensor):
    results = model(input_tensor).argmax(1).item()
    return results
    # return ','.join(results)


def label(value: Union[int, float]):
    return primary_mappings.get(int(value), 'Mapping not found')


def classify(img):
    input_tensor = img_to_tensor(img)
    classification = classify_to_num(input_tensor)
    return label(classification)


def classify_extractions_dictionary(images: dict):
    return {coordinates: classify(png)
            for (coordinates, png) in images.items()}


def classify_page(
        page: Image,
        coordinates: Union[list, tuple]):

    images = extract_page_images(page, coordinates)
    return classify_extractions_dictionary(images)
