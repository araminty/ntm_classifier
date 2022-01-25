import numpy as np
from PIL import Image
from typing import Union

from ntm_classifier.extract import alt_str_format

# Reminder the pdfminer6 is giving me the xmls


def png_buffer_from_filepath(filepath):
    with Image.open(filepath) as img:
        img = img.crop((0, 0, img.size[0], img.size[1]))
        return img


def tree_extract_xml_figures(xml_tree, page_number):
    page = xml_tree.getroot()[page_number]
    return page_extract_xml_figures(page)


def page_extract_xml_figures(page):
    def gen():
        for element in page:
            if element.tag == 'figure':
                # yield {'page':page_number, element.get('bbox')}
                yield element

    return list(gen())


def tree_get_textbox_locations_list(xml_tree, page_number):
    page = xml_tree[page_number]
    return page_get_textbox_locations_list(page)


def page_get_textbox_locations_list(page):
    def gen():
        for element in page:
            if element.tag == 'text':
                yield element.get('bbox', '')
    return list(gen())


def whiteout_box(
        image: Union[np.ndarray, Image.Image],
        bbox: Union[str, tuple],
        page_bb: Union[str, tuple] = "0.000,0.000,595.320,841.920",
        as_array: Union[None, bool] = None,
        raise_error: bool = False,
        invert_color: bool = False):
    as_array = as_array or not isinstance(image, Image.Image)
    matrix = None
    if isinstance(image, Image.Image):
        matrix = np.asarray(image)
    elif isinstance(image, np.ndarray):
        matrix = image
    if not isinstance(matrix, np.ndarray):
        if raise_error:
            raise ValueError(
                "Image passed to whiteout_box was not PIL Image or numpy array")
        return image

    (x1, y1), (x2, y2) = alt_str_format(bbox)
    (_, _), (px, py) = alt_str_format(page_bb)

    width = matrix.shape[1]
    height = matrix.shape[0]

    x1 = max((0, int(round(width * x1 / px))))
    # y1 = max((0, int(round(height*y1/py))))
    x2 = min((width, int(round(width * x2 / px))))
    # y2 = min((height, int(round(height*y2/py))))
    # x2 = width-max((0, int(round(width*x1/px))))
    y1 = height - max((0, int(round(height * y1 / py))))
    # x1 = width-min((width, int(round(width*x2/px))))
    y2 = height - min((height, int(round(height * y2 / py))))

    print(x1, x2, y1, x2)

    if invert_color:
        matrix[y2:y1, x1:x2] = 0
    else:
        matrix[y2:y1, x1:x2] = 255

    if as_array is True:
        return matrix
    return Image.fromarray(matrix)


def whiteout_page_text(
        page: Union[np.ndarray, Image.Image],
        xml):
    pass


# def extract_image_with_textbox_whiteout():
#     pass
