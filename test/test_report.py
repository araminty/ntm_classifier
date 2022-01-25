import unittest
import numpy as np

from matplotlib.figure import Figure

from ntm_classifier.report import (
    get_heatmap,
    primary_report,
    load_classification_table,
    primary_mappings
)


class TestReport(unittest.TestCase):

    def test_make_heatmap(self):
        # Create random data to serve as a test case
        random_array = np.random.randint(20, size=(4, 4))
        labels = ['dummy', 'label', 'values', 'testing']
        # Make a sample heatmap
        heatmap = get_heatmap(random_array, labels=labels)
        # Assert it is a matplotlib image to test 
        # if it successfully created the image
        self.assertIsInstance(heatmap, Figure)

    def test_make_table(self):
        # Load a few random elements from the test set to check
        df = load_classification_table().sample(100)
        labels = list((p.lower() for p in primary_mappings.values()))
        # Create a confusion matrix
        matrix = primary_report(df, labels)
        # Assert it is a numpy array to test if
        # it successfully created the confusion matrix
        self.assertIsInstance(matrix, (np.ndarray,))


if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover
