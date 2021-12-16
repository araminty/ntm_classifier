import unittest

from ntm_classifier.extract import (
    verify_coordinates,
    extract_image,
    classify_by_coordinates)
from ntm_classifier.load_resources import (
    load_test_page,
    load_test_slice,)
from ntm_classifier.classifier import primary_mappings


class TestCoordinateCheck(unittest.TestCase):
    test_page_png = load_test_page()

    def test_verify_int_string(self):
        page = self.test_page_png
        test = verify_coordinates(page=page, coordinates="(154, 234)")
        self.assertEqual(test, (154, 234))

    def test_verify_float_string(self):
        page = self.test_page_png
        self.assertEqual((page.width, page.height), (2481, 3508))
        test = verify_coordinates(page=page, coordinates="(.2, .3)")
        self.assertEqual(test, (496, 1052))

    # def test_invalid_coordinate(self):
    #     # need cases to put here...
    #     pass


class TestExtract(unittest.TestCase):
    test_page_png = load_test_page()
    test_slice = load_test_slice().crop((0, 0, 200, 180))

    def test_extract_float(self):
        page = self.test_page_png
        test_slice = self.test_slice
        result = extract_image(page, (.1713, .2993), (.2519, .3506))
        self.assertEqual(result, test_slice)

        result = extract_image(page, "(.1713, .2993)", "(.2519, .3506)")
        self.assertEqual(result, test_slice)

    def test_extract_int(self):
        page = self.test_page_png
        test_slice = self.test_slice
        result = extract_image(page, (425, 1050), (625, 1230))
        self.assertEqual(result, test_slice)

        result = extract_image(page, "(425, 1050)", "(625, 1230)")
        self.assertEqual(result, test_slice)


class TestClassifyByCoordinates(unittest.TestCase):
    test_page_png = load_test_page()

    def test_classify_by_coordinates(self):
        page = self.test_page_png
        result = classify_by_coordinates(page, "(425, 1050)", "(625, 1230)")
        self.assertIsInstance(result, str)
        self.assertIn(result, primary_mappings.values())

class TestClassifyFullPage(unittest.Testcase):
    test_page_png = load_test_page()


if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover
