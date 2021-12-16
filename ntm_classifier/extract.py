"""
This file is for extracting the individual images from a page so they can be passed
to the classifier.  The coordinates need to already be found.
The page should be passed as a png image.  The coordinates should be passed as pixel
coordinates and x-y format.
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


def extract_image(
    page: Image,
    upper_left: Union[str, tuple],
    lower_right: Union[str, tuple],):
    upper_left = verify_coordinates(page, upper_left)
    lower_right = verify_coordinates(page, lower_right)

    bbox = upper_left[0], upper_left[1], lower_right[0], lower_right[1]
    return page.crop(bbox)


def classify_by_coordinates(
    page: Image,
    upper_left: Union[str, tuple],
    lower_right: Union[str, tuple],):
    png = extract_image(page, upper_left, lower_right)
    return classify(png)

def page_classifications_by_coordinates(
    page: Image,
    coordinates: Union[list, tuple]):
    pass