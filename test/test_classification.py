import unittest

from torch import Tensor


from ntm_classifier.preprocess import img_to_tensor
from ntm_classifier.classifier import (
    classify_to_num,
    label,
    classify,
    primary_mappings,
)
from ntm_classifier.load_resources import load_test_image
from ntm_classifier.load_resources import load_test_tensor


class TestClassify(unittest.TestCase):
    test_image = load_test_image()
    test_array = load_test_tensor()

    def test_convert_img_to_tensor(self):
        test_image = self.test_image
        result = img_to_tensor(test_image)
        self.assertIsInstance(result, Tensor)

    def test_classify_tensor_to_int(self):
        result = classify_to_num(self.test_array)
        self.assertIsInstance(result, int)

    def test_convert_int_to_str(self):
        result = label(1)
        self.assertIn(1, primary_mappings.keys())
        self.assertIsInstance(result, str)
        self.assertIn(result, primary_mappings.values())

    def test_classify_png_to_str(self):
        result = classify(self.test_image)
        self.assertIsInstance(result, str)
        self.assertIn(result, primary_mappings.values())


if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover
