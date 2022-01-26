from typing import Union
import pandas as pd
from pkg_resources import resource_filename
from ntm_classifier.load_resources import process_mappings_group
from ntm_classifier.load_resources import load_classification_table

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


def build_tag_mappings_from_group(
        df: Union[pd.DataFrame, None],
        group: str = 'primary',
        tag_list_column_name: str = 'tags',
        file_column_name: str = 'file',
        store=False):

    if df is None:
        df = load_classification_table('tags.csv')

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
