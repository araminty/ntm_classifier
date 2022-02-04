import json
# import pickle
import torch
import torchvision

from collections import OrderedDict  # noqa
from pkg_resources import resource_string, resource_filename
# from keras.preprocessing.image import load_img
from PIL import Image
import xml.etree.ElementTree as ET
from importlib.resources import path as ir_path


def get_image_dir_path(name='test_image_dir'):
    with ir_path('ntm_data.test_files', name) as path_obj:
        path = str(path_obj)
    return path


def load_xml_test(name='second_xml.xml'):
    filename = resource_filename('ntm_data.test_files', name)
    with open(filename) as file:
        tree = ET.parse(file)
    return tree.getroot()


def load_text_lines_list(name='uncombined_bboxes.txt'):
    filename = resource_filename('ntm_data.test_files', name)
    with open(filename) as file:
        lines = file.readlines()
    return [line.strip('\n') for line in lines]


def save_model_torch_script(
        model,
        input_dimensions=(5, 3, 224, 224),
        name='unnamed_model.pt'):
    filepath = resource_filename('ntm_data', name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sample_shape = torch.rand(size=input_dimensions).to(device)
    model = torch.jit.trace(model, sample_shape)
    torch.jit.save(model, filepath)


def load_base_model():
    resnet50 = torch.hub.load(  # noqa
        'NVIDIA/DeepLearningExamples:torchhub',
        # num_classes=13,
        'nvidia_resnet50',
        pretrained=True,
    )

    return resnet50
    # model_path = resource_filename('ntm_data', 'base_resnet50.pt')
    # return torch.load(model_path)


def load_model(state_dict_name="primary_state_dict.pt"):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    state_dict_path = resource_filename('ntm_data', state_dict_name)

    state_dict = torch.load(state_dict_path)
    n_outputs = state_dict.get('fc.weight').shape[0]
    fc_in = state_dict.get('fc.weight').shape[1]

    resnet = torchvision.models.resnet50()
    model = torch.nn.Sequential(OrderedDict({
        'resnet': resnet,
        'fc': torch.nn.Linear(fc_in, n_outputs),
        'output': torch.nn.LogSoftmax(dim=1)
    }))
    model.load_state_dict(state_dict)
    return model.to(device)


def load_mappings():
    resource = resource_string('ntm_data', 'mappings.json')
    return json.loads(resource.decode('utf-8'))


def process_mappings_group(group='primary'):
    mappings = load_mappings()
    group_mappings = mappings.get(group, {})
    group_mappings = {int(k): v for (k, v) in group_mappings.items()}
    return group_mappings


def lowercase_inverted_mappings(group='primary'):
    group_mappings = process_mappings_group(group)
    return {v.lower(): k for (k, v) in group_mappings.items()}


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


def load_test_page(filename='sample_page.png'):
    img_path = resource_filename('ntm_data.test_files', filename)
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
