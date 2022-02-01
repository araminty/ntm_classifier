import unittest
import numpy as np
# from PIL import Image

from ntm_classifier.load_resources import (
    load_test_page,
    load_test_whiteout_crop,
)
from ntm_classifier.image_from_document import (
    whiteout_box,
    whiteout_page_text,
    combine_adjacent_bboxes,)
from ntm_classifier.load_resources import (
    load_xml_test, load_text_lines_list)

from ntm_classifier.image_from_document import (
    page_get_textbox_locations_list)


class TestWhiteOut(unittest.TestCase):

    def test_combine_adjacent_bboxes(self):
        text_lines = load_text_lines_list(
            name='uncombined_bboxes.txt')[:20]

        target = ['298.010,790.620,309.725,805.620',
                  '205.130,769.500,280.025,784.500',
                  '285.020,769.500,308.345,784.500',
                  '313.340,769.500,327.500,784.500',
                  '327.620,769.500,402.545,784.500',
                  '300.890,27.528,306.890,39.528']

        combined = combine_adjacent_bboxes(text_lines, gap=0.1)
        self.assertEqual(combined, target)

    def test_redact_page_text_from_xml(self):
        page = load_test_page()
        test_xml = load_xml_test()[0]
        target = load_test_page('redacted.png')

        redacted = whiteout_page_text(page, test_xml, invert_color=True)
        self.assertEqual(redacted, target)

    def test_redact_wider_gap(self):
        page = load_test_page()
        test_xml = load_xml_test()[0]
        target = load_test_page('redacted2.png')

        redacted = whiteout_page_text(page, test_xml, invert_color=True, gap=5)
        self.assertEqual(redacted, target)

    def test_blank_entire_page(self):
        page = load_test_page()

        coords = f"0,0,{page.width},{page.height}"

        covered = whiteout_box(page, coords, page_bb=coords, invert_color=True)
        covered = np.asarray(covered)
        self.assertEqual(covered.sum(), 0)

    def test_blank_single_image(self):
        page = load_test_page().crop((300, 1000, 700, 1500))
        target = load_test_whiteout_crop()
        coords = "125,250,325,430"
        page_bb = "0,0,400,500"
        covered = whiteout_box(page, coords, page_bb=page_bb, as_array=False)
        self.assertEqual(covered, target)


if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover
