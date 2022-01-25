import unittest

from ntm_classifier.extract import (
    verify_coordinates,
    extract_image,
    extract_page_images,
)
from ntm_classifier.load_resources import (
    load_test_page,
    load_test_crop,
)
from ntm_classifier.classifier import (
    primary_mappings,
    classify_extractions_dictionary,
    classify_page,
)


class TestCoordinateCheck(unittest.TestCase):
    test_page_png = load_test_page()

    def test_verify_int_tuple(self):
        page = self.test_page_png
        test = verify_coordinates(page=page, coordinates=(154, 234))
        self.assertEqual(test, (154, 234))

    def test_verify_int_string(self):
        page = self.test_page_png
        test = verify_coordinates(page=page, coordinates="(154, 234)")
        self.assertEqual(test, (154, 234))

    def test_verify_float_tuple(self):
        page = self.test_page_png
        self.assertEqual((page.width, page.height), (2481, 3508))
        test = verify_coordinates(page=page, coordinates=(.2, .3))
        self.assertEqual(test, (496, 1052))

    # Not currently in use so not putting priority on getting this working
    # def test_verify_float_string(self):
    #     page = self.test_page_png
    #     self.assertEqual((page.width, page.height), (2481, 3508))
    #     test = verify_coordinates(page=page, coordinates="(.2, .3)")
    #     self.assertEqual(test, (496, 1052))

    # Some cases of examples that should fail
    # would flesh out the test suite more
    # def test_invalid_coordinate(self):
    #     pass


class TestExtract(unittest.TestCase):
    test_page_png = load_test_page()
    test_slice = load_test_crop().crop((0, 0, 200, 180))

    def test_extract_float(self):
        page = self.test_page_png
        test_slice = self.test_slice
        result = extract_image(page, (.1713, .1710), (.2519, .2223))
        self.assertEqual(result, test_slice)

    # Not currently in use so not putting priority on getting this working
    # def test_extract_float_string(self):
    #     page = self.test_page_png
    #     test_slice = self.test_slice
    #     result = extract_image(page, "(.1713, .1710)", "(.2519, .2223)")
    #     self.assertEqual(result, test_slice)

    def test_extract_int(self):
        page = self.test_page_png
        test_slice = self.test_slice
        result = extract_image(page, (425, 600), (625, 780))
        self.assertEqual(result, test_slice)
        # self.assertEqual(np.asarray(result), np.asarray(test_slice))

        result = extract_image(page, "(425, 600)", "(625, 780)")
        # self.assertEqual(np.asarray(result), np.asarray(test_slice))
        self.assertEqual(result, test_slice)


class TestFullPage(unittest.TestCase):
    test_page_png = load_test_page()
    test_page_coordinates = [(
        (425, 600), (625, 780)),
        ((425, 770), (625, 950)),
        ((425, 1050), (625, 1230)),
        ((425, 1220), (625, 1400)),
        ((425, 1500), (625, 1680)),
        ((425, 1675), (625, 1855)),
        ((425, 1960), (625, 2140)),
        ((425, 2130), (625, 2310)),
        ((425, 2440), (625, 2620)),
        ((425, 2610), (625, 2790))]

    def test_extract_full_page(self):
        extractions = extract_page_images(
            self.test_page_png,
            self.test_page_coordinates,
        )

        for c, png in extractions.items():
            filename = "{},{}_{},{}.png".format(
                c[0][0],
                c[0][1],
                c[1][0],
                c[1][1])
            target = load_test_crop(filename).crop((0, 0, 200, 180))
            self.assertEqual(png, target)

    def test_classify_page(self):
        extractions = extract_page_images(
            self.test_page_png,
            self.test_page_coordinates,
        )
        results = classify_extractions_dictionary(extractions)
        for result in results.values():
            self.assertIn(result, primary_mappings.values())

        results2 = classify_page(
            self.test_page_png, self.test_page_coordinates)

        self.assertEqual(sorted(results.items()), sorted(results2.items()))

if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover
