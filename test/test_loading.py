import unittest
from torch.nn.modules import Sequential
from torch import Tensor
# from numpy import ndarray


from ntm_classifier.load_resources import (
    load_model,
    load_mappings,
    load_test_image,
    load_test_tensor,
    load_test_page,
    load_test_crop,
    load_classification_table,
)

from PIL.Image import Image


"""The package includes data for the models, files for the confusion matrix
report and test case data.  This file serves to check that they
successfully load."""


class TestLoading(unittest.TestCase):

    def test_load_primary_model(self):
        primary = load_model()
        self.assertIsInstance(primary, Sequential)

    def test_load_mappings(self):
        self.assertIsInstance(load_mappings(), dict)

    def test_load_sample_image(self):
        self.assertIsInstance(load_test_image(), Image)

    def test_load_sample_page(self):
        self.assertIsInstance(load_test_page(), Image)

    def test_load_sample_crop(self):
        self.assertIsInstance(load_test_crop(), Image)

    def test_load_sample_tensor(self):
        self.assertIsInstance(load_test_tensor(), Tensor)

    def test_load_class_table(self):
        from pandas import DataFrame
        self.assertIsInstance(load_classification_table(), DataFrame)


if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover
