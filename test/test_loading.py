import unittest
from torch.nn.modules import Sequential
from torch import Tensor
# from numpy import ndarray


from ntm_classifier.load_resources import (
    load_primary,
    load_mappings,
    load_test_image,
    load_test_tensor,
    load_test_page,
    load_test_slice,
)

from PIL.Image import Image


class TestLoading(unittest.TestCase):

    def test_load_primary_model(self):
        primary = load_primary()
        self.assertIsInstance(primary, Sequential)

    def test_load_mappings(self):
        self.assertIsInstance(load_mappings(), dict)

    def test_load_sample_image(self):
        self.assertIsInstance(load_test_image(), Image)

    def test_load_sample_page(self):
        self.assertIsInstance(load_test_page(), Image)

    def test_load_sample_slice(self):
        self.assertIsInstance(load_test_slice(), Image)

    def test_load_sample_tensor(self):
        self.assertIsInstance(load_test_tensor(), Tensor)



if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover