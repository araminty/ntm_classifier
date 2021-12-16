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
    return json.loads(
        resource_string(
            'ntm_data',
            'mappings.json').decode('utf-8'))


def load_mappings_reverse():
    mappings = load_mappings()
    return {l: {v: k for (k, v) in m.items()} for (l, m) in mappings.items()}


def load_test_image():
    img_path = resource_filename('ntm_data.test_files', 'lighted.png')
    return Image.open(img_path)


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
    tensor_path = resource_filename('ntm_data.test_files', 'sample_page.png')
    return Image.open(tensor_path)


def load_test_crop(filename='425,600_625,780.png'):
    tensor_path = resource_filename(
        'ntm_data.test_files',
        f"test_crops/{filename}")
    return Image.open(tensor_path)
