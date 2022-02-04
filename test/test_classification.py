import unittest

from torch import Tensor


from ntm_classifier.preprocess import img_to_tensor
from ntm_classifier.classifier import (
    classify_to_num,
    label,
    classify,
    primary_mappings,
    classify_page_from_xml,
    classify_directory,
)
from ntm_classifier.load_resources import (
    get_image_dir_path,
    load_test_image,
    load_test_tensor,
    load_test_page,
    load_xml_test)


class TestDirClassify(unittest.TestCase):
    def test_classify_directory(self):
        dir_path = get_image_dir_path()
        classifications = classify_directory(dir_path)
        self.assertIsInstance(classifications, dict)


class TestPageClassify(unittest.TestCase):
    def test_classify_page(self):
        page = load_test_page()
        xml = load_xml_test()[0]

        bbox_cls_dict = classify_page_from_xml(page, xml)

        for k, v in bbox_cls_dict.items():
            self.assertEqual(len(k.split(',')), 4)
            self.assertIn(v, primary_mappings.values())


class TestClassify(unittest.TestCase):
    

    def test_convert_img_to_tensor(self):
        test_image = load_test_image()
        result = img_to_tensor(test_image)
        self.assertIsInstance(result, Tensor)

    def test_classify_tensor_to_int(self):
        test_array = load_test_tensor()
        result = classify_to_num(test_array)
        self.assertIsInstance(result, int)

    def test_convert_int_to_str(self):
        result = label(1)
        self.assertIn(1, primary_mappings.keys())
        self.assertIsInstance(result, str)
        self.assertIn(result, primary_mappings.values())

    def test_classify_png_to_str(self):
        test_image = load_test_image()
        result = classify(test_image)
        self.assertIsInstance(result, str)
        self.assertIn(result, primary_mappings.values())


if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover
