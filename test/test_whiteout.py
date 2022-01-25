import unittest
import numpy as np
from PIL import Image

from ntm_classifier.load_resources import (
    load_test_page,
    load_test_whiteout_crop,
)
from ntm_classifier.image_from_document import whiteout_box


class TestWhiteOut(unittest.TestCase):
    test_page = load_test_page()

    def test_blank_entire_page(self):
        page = self.test_page.copy()

        coords = f"0,0,{page.width},{page.height}"

        covered = whiteout_box(page, coords, page_bb=coords, invert_color=True)
        covered = np.asarray(covered)
        self.assertEqual(covered.sum(), 0)

    def test_blank_single_image(self):
        page = self.test_page.crop((300, 1000, 700, 1500))
        target = load_test_whiteout_crop()
        coords = "125,50,325,230"
        page_bb = "0,0,400,500"
        covered = whiteout_box(page, coords, page_bb=page_bb, as_array=False)
        self.assertEqual(covered, target)
    


if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover
