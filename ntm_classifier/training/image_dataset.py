from typing import Union

import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from torchvision import transforms
# from PIL import Image

from ntm_classifier.load_resources import load_report_image
from ntm_classifier.load_resources import load_classification_table
from ntm_classifier.load_resources import lowercase_inverted_mappings
from ntm_classifier.preprocess import preprocess as image_preprocess
# from ntm_classifier.check_tqdm import tqdm_check

# if tqdm_check():
#     from tqdm import tqdm as tqdm_c


# image_preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.PILToTensor(),
#     transforms.ConvertImageDtype(torch.float),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
# ])

tcType = transforms.Compose


class ImageDataset(Dataset):
    def __init__(self,
                 df: Union[DataFrame, str, None] = 'primary_tags.csv',
                 label_column: str = "primary",
                 output_labels: str = "primary",
                 image_file_column: str = "file",
                 transform: Union[tcType, None] = image_preprocess,
                 limit_n=None):
        if df is None:
            df = load_classification_table('primary_tags.csv')
        if isinstance(df, str):
            df = load_classification_table(df)
        if limit_n is not None:
            df = df.head(limit_n)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.filenames = df[image_file_column]
        self.y = df[label_column]
        self.transform = transform
        self.output_map = lowercase_inverted_mappings(output_labels)

        # self.inputs = self.load_images()
        self.outputs = self.map_outputs()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # image = self.images[idx]
        # if self.transform is not None:
        #     if isinstance(image, Image.Image):
        #         image = self.transform(image)
        #     else:
        #         image = torch.stack(list(map(self.transform, image)))

        image = self.load_image_idx(idx)

        sample = {'image': image, 'result': self.outputs[idx]}

        return sample

    def map_outputs(self):
        output_labels = self.y.str.lower().apply(self.output_map.get)
        self.na_value = output_labels.max()+1.0
        output_labels = output_labels.fillna(self.na_value)
        return torch.from_numpy(output_labels.values).to(self.device).long()

    # def load_images(self):
    #     fn = self.filenames
    #     if tqdm_check():
    #         tqdm_c.pandas(desc="Loading images")
    #         images = fn.progress_apply(load_report_image)
    #     else:
    #         images = fn.apply(load_report_image)

    #     self.images = images
    #     return torch.stack(list(map(self.transform, images))).to(self.device)

    def load_image_idx(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filenames = self.filenames.values[idx]
        if isinstance(filenames, str):
            image = load_report_image(filenames)
            return self.transform(image).unsqueeze(0).to(self.device)
        else:
            images = map(load_report_image, filenames)
            return torch.stack(
                list(map(self.transform, images))).to(self.device)
        # images = filenames.apply(load_report_image).values
