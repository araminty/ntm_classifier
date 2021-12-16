"""
This file is for extracting the individual images from a page
so they can be passed to the classifier.  The coordinates need
to already be found.  The page should be passed as a png image.
The coordinates should be passed as pixel coordinates and x-y
format.
"""

from ntm_classifier.classifier import classify
from typing import Union
from PIL.Image import Image


def _convert_str_float(coordinates: str):
    coords = coordinates.split(',')
    coords = (c.strip('(').strip(')') for c in coords)
    return tuple(float(c) for c in coords if c.isnumeric)


def verify_coordinates(page: Image, coordinates: Union[str, tuple]):
    if not isinstance(coordinates, (str, tuple)):
        raise TypeError(
            f"Coordinates: {coordinates} must be passed as str or tuple")

    def process_coordinate_pair(coordinate: tuple):
        w, h = coordinate
        if (w == 0 or w > 1.0) or (h == 0 or h > 1.0):
            return min(round(w), page.width), min(round(h), page.height)
        return round(w * page.width), round(h * page.height)

    if isinstance(coordinates, str):
        coordinates = _convert_str_float(coordinates)

    return process_coordinate_pair(coordinates)


def verify_coordinate_pair_from_str(page: Image, pair_str: str):
    splits = pair_str.split(',')
    if len(splits) == 4:
        one = ','.join(splits[0:2]).strip()
        two = ','.join(splits[2:4]).strip()
        return verify_coordinates(page, one), verify_coordinates(page, two)
    else:
        raise ValueError(f"""Coordinates must be passed as a pair
        each in the format: (x, y)
        Or a single value in the format: (x, y), (x, y)
        Received {pair_str}""")


def extract_image(
        page: Image,
        upper_left: Union[str, tuple],
        lower_right: Union[str, tuple] = None,):
    if lower_right is None and isinstance(upper_left, str):
        verify_coordinate_pair_from_str(page, upper_left)
    if lower_right is None and isinstance(upper_left, tuple):
        if (len(upper_right) == 2 and
            isinstance(upper_left[0], tuple) and
                isinstance(upper_left[1], tuple)):
            upper_left, lower_right = upper_left
        else:
            raise ValueError(f"""Coordinates must be passed as a pair
            each in the format: (x, y)
            Or a single value in the format: (x, y), (x, y)
            Received {upper_left} and {lower_right}""")

    upper_left = verify_coordinates(page, upper_left)
    lower_right = verify_coordinates(page, lower_right)

    bbox = upper_left[0], upper_left[1], lower_right[0], lower_right[1]
    return page.crop(bbox)


def extract_page_images(
        page: Image,
        coordinates: Union[list, tuple]):

    extractions = {}
    for coordinate in coordinates:
        upper_left = coordinate[0]
        lower_right = coordinate[1]
        extractions[coordinate] = extract_image(
            page, upper_left, lower_right)

    return extractions


def classify_extractions_dictionary(images: dict):
    return {coordinates: classify(png)
            for (coordinates, png) in images.items()}


def classify_page(
        page: Image,
        coordinates: Union[list, tuple]):

    images = extract_page_images(page, coordinates)
    return classify_extractions_dictionary(images)
