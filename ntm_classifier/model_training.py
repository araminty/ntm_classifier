import torch
import pandas as pd

from pkg_resources import resource_filename
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


def build_tag_mappings_from_group(
        df: pd.DataFrame,
        group: str = 'primary',
        tag_list_column_name: str = 'tags',
        file_column_name: str = 'file',
        store=False):

    result_df = df[[file_column_name, tag_list_column_name]].copy()
    # result_df[file_column_name] = df[file_column_name]

    mappings = process_mappings_group(group)
    lowercases = {v.lower(): k for (k, v) in mappings.items()}

    def get_first_matching_label(tags_str):
        for label in lowercases:
            if label in tags_str.lower():
                return label
        return None

    if mappings == {}:
        raise ValueError(f"{group} must be group from mappings "
                         "json with at least one item")

    name = (group if group not in (tag_list_column_name,
                                   file_column_name) else 'label')

    if use_tqdm:
        tqdm.pandas(desc='building label row')
        result_df[name] = df['tags'].progress_apply(get_first_matching_label)
    else:
        result_df[name] = df['tags'].apply(get_first_matching_label)

    sort_index = df[file_column_name].str.strip('.png').\
        apply(int).sort_values().index

    result_df = result_df.loc[sort_index].reset_index(drop=True)

    if store:
        filepath = resource_filename(
            'ntm_data.table_data', f"{name}_tags.csv")
        result_df.to_csv(filepath, index=False)

    return result_df


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

    sort_index = df['file'].str.strip('.png').apply(int).sort_values().index

    df = df.loc[sort_index].reset_index(drop=True)

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
