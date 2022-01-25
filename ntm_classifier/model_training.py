import torch

from datasets import Dataset
# from numpy import argmax

from ntm_classifier.load_resources import (
    # load_primary,
    process_mappings_group,
    load_classification_table,
    load_report_image,
)
# from ntm_classifier.preprocess import img_to_tensor
from ntm_classifier.preprocess import preprocess

try:
    get_ipython().__class__.__name__  # noqa
    use_tqdm = True
    from tqdm import tqdm
except BaseException:
    if __name__ == '__main__':
        use_tqdm = True
        from tqdm import tqdm
    else:
        use_tqdm = False


# model = load_primary()


primary_mappings = process_mappings_group('primary')
primary_mappings_lower = tuple(v.lower() for v in primary_mappings.values())
# primary labels reversed lowercase
dict_pmrl = {v.lower(): k for (k, v) in primary_mappings.items()}


def get_y_int(label): return dict_pmrl.get(label.lower(), label)


def get_primary_if_any(tags):
    for m in primary_mappings_lower:
        if m in tags.lower():
            return m
    return None


# import numpy as np
def ready_dataframe():

    df = load_classification_table()
    if use_tqdm:
        tqdm.pandas(desc='building primary label row')
        df['primary'] = df['tags'].progress_apply(get_primary_if_any)
    else:
        df['primary'] = df['tags'].apply(get_primary_if_any)
    df['primary'] = df['primary'].fillna('None')
    if use_tqdm:
        tqdm.pandas(desc='loading images')
        df['image'] = df['file'].progress_apply(load_report_image)
    else:
        df['image'] = df['file'].apply(load_report_image)

    return df[['primary', 'image']]

    # df['array'] = df['image'].apply(np.array)
    # return df[['primary', 'image', 'array']]

    # if use_tqdm:
    #     tqdm.pandas(desc='tensorizing images')
    #     df['tensor'] = df['image'].progress_apply(img_to_tensor)
    # else:
    #     df['tensor'] = df['image'].apply(img_to_tensor)

    # df['tensor'] = df['tensor'].apply(lambda x: x.unsqueeze(1))

 

class CustomDataset(Dataset):

    def __init__(
            self,
            df,
            label_column: str = 'primary',
            # tensor_column: str = 'tensor',
            image_column: str = 'image',
            ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # X = df[tensor_column].values
        X = df[image_column].values

        if use_tqdm:
            y = df[label_column].progress_apply(get_y_int).values
        else:
            y = df[label_column].apply(get_y_int).values
        y = torch.from_numpy(y).to(device).long()

        self.X = X
        self.y = y
        self.length = len(X)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X = self.X[idx]
        X = preprocess(X)
        y = self.y[idx]

        return X, y

# images_dataset = CustomDataset(ready_dataframe())
# dl = torch.utils.data.DataLoader(images_dataset)