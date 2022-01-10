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
        random_array = np.random.randint(20, size=(4,4))
        labels = ['foo', 'bar', 'fizz', 'buzz']
        heatmap = get_heatmap(random_array, labels=labels)
        self.assertIsInstance(heatmap, Figure)

    def test_make_table(self):
        df = load_classification_table().sample(20)
        labels = list((p.lower() for p in primary_mappings.values()))
        matrix = primary_report(df, labels)

        self.assertIsInstance(matrix, (np.ndarray,))
        



if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover
