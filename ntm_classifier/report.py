import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


from ntm_classifier.load_resources import (
    load_classification_table,
    load_report_image,
    load_test_crop,
)

from ntm_classifier.classifier import (
    classify,
    primary_mappings,
)
from tqdm import tqdm

primary_mappings_lower = tuple(v.lower() for v in primary_mappings.values())


def get_primary_if_any(tags):
    for m in primary_mappings_lower:
        if m in tags.lower():
            return m
    return None


def get_heatmap(
        matrix,
        labels,
        title="Primary Labels Confusion Matrix"):
    fig, ax = plt.subplots(facecolor='white')
    im = ax.imshow(matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    return fig


def primary_report(df, labels):

    tqdm.pandas(desc='building primary label row')
    df['primary'] = df['tags'].progress_apply(get_primary_if_any)

    tqdm.pandas(desc='loading images')
    df['image'] = df['file'].progress_apply(load_report_image)

    tqdm.pandas(desc='running_classifier')
    df['results'] = df['image'].progress_apply(classify).str.lower()

    results_df = df[['primary', 'results']].dropna()

    y1, y2 = results_df['primary'], results_df['results']
    primary_confusion_matrix = confusion_matrix(y1, y2, labels=labels)

    return primary_confusion_matrix


if __name__ == '__main__':
    df = load_classification_table()
    labels = list((p.lower() for p in primary_mappings.values()))
    matrix = primary_report(df, labels)
    heat = get_heatmap(matrix, labels)
    heat.savefig('heatmap.png', format='png')
