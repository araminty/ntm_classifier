import json
# import pickle
import torch

from collections import OrderedDict
from pkg_resources import resource_string, resource_filename
# from keras.preprocessing.image import load_img
from PIL import Image


def load_primary():

    resnet50 = torch.hub.load(
        'pytorch/vision:v0.10.0',
        'resnet18',
        pretrained=False)

    model = torch.nn.Sequential(OrderedDict({
        'resnet': resnet50,
        'fc': torch.nn.Linear(1000, 13),
        # 'categories': nn.LogSoftmax(dim=13),
        'output': torch.nn.LogSoftmax(dim=1)
    }))

    # weights = torch.load(resource_stream('ntm_data', 'state_dict.pt'))
    weights_path = resource_filename('ntm_data', 'state_dict.pt')
    weights = torch.load(weights_path)

    try:
        model.load_state_dict(weights)
    except RuntimeError:
        # TODO: This obviously needs to get removed
        # but I'm still playing around with the exact model dimensions
        print("model size mismatch")

    return model


def load_mappings():
    resource = resource_string('ntm_data', 'mappings.json')
    return json.loads(resource.decode('utf-8'))


def load_mappings_reverse():
    mappings = load_mappings()
    return {l: {v: k for (k, v) in m.items()} for (l, m) in mappings.items()}


def load_test_image():
    img_path = resource_filename('ntm_data.test_files', 'lighted.png')
    with Image.open(img_path) as img:
        img = img.crop((0, 0, img.size[0], img.size[1]))
        return img


# def load_test_array():
#     array_path = resource_filename('ntm_data.test_files', 'sample_array.pkl')
#     with open(array_path, 'rb') as file:
#         result = pickle.load(file)
#     return result


def load_test_tensor():
    tensor_path = resource_filename('ntm_data.test_files', 'sample_tensor.pt')
    with open(tensor_path, 'rb') as file:
        result = torch.load(file)
    return result


def load_test_page():
    img_path = resource_filename('ntm_data.test_files', 'sample_page.png')
    with Image.open(img_path) as img:
        img = img.crop((0, 0, img.size[0], img.size[1]))
        return img


def load_test_whiteout_crop():
    img_path = resource_filename('ntm_data.test_files',
                                 'covered_test_example.png')
    with Image.open(img_path) as img:
        img = img.crop((0, 0, img.size[0], img.size[1]))
        return img


def load_test_crop(filename='425,600_625,780.png'):
    img_path = resource_filename('ntm_data.test_files.test_crops', filename)
    with Image.open(img_path) as img:
        img = img.crop((0, 0, img.size[0], img.size[1]))
        return img


def load_classification_table(filename='tags.csv'):
    import pandas as pd
    data_path = resource_filename('ntm_data.table_data', filename)
    return pd.read_csv(data_path)


def load_report_image(filename='0.png'):
    img_path = resource_filename('ntm_data.table_data.images', filename)
    with Image.open(img_path) as img:
        img = img.crop((0, 0, img.size[0], img.size[1]))
        return img
