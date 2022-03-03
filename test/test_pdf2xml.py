import unittest
import os
import shutil

from ntm_classifier.utils_e2.pdf2xml import (
    get_xml_tmp_path,
    pdf_to_xml_file,
    pdf_to_xml_object,
    make_page_images,
)
from ntm_classifier.load_resources import get_test_pdf_path


class TestPDF2XML(unittest.TestCase):

    def test_get_pdf_images(self):
        pdf_path = get_test_pdf_path()
        make_page_images(pdf_path)
        images_path = os.path.join(pdf_path[:-4], 'page_images_high_res')
        image_names = os.listdir(images_path)
        shutil.rmtree(pdf_path[:-4])
        self.assertTrue(len(image_names) > 0)

    def test_get_test_obj_path(self):
        pdf_path = get_test_pdf_path()
        pdf_path_end = '/'.join(pdf_path.split('/')[-3:])
        target = 'ntm_data/test_files/1964674_LV_01_2022.pdf'
        self.assertEqual(pdf_path_end, target)

    def test_convert_file_to_file(self):
        pdf_path = get_test_pdf_path()
        xml_path = get_xml_tmp_path(pdf_path)
        already_exists = os.path.exists(xml_path)
        in_test_path = "ntm_data/test_files" in xml_path
        self.assertFalse(already_exists)
        self.assertTrue(in_test_path)

        pdf_to_xml_file(pdf_path, xml_path)

        self.assertTrue(os.path.exists(xml_path))
        if (not already_exists) and (in_test_path):
            os.remove(xml_path)

    def test_buffer_to_file(self):
        pdf_path = get_test_pdf_path()
        xml_path = get_xml_tmp_path(pdf_path)
        already_exists = os.path.exists(xml_path)
        in_test_path = "ntm_data/test_files" in xml_path
        self.assertFalse(already_exists)
        self.assertTrue(in_test_path)

        with open(pdf_path, 'rb') as pdf_buffer:
            pdf_to_xml_file(pdf_buffer, xml_path)

        self.assertTrue(os.path.exists(xml_path))
        if (not already_exists) and (in_test_path):
            os.remove(xml_path)

    def test_convert_file_to_xml_object(self):
        pdf_path = get_test_pdf_path()
        xml_tuple = pdf_to_xml_object(pdf_path)
        text = tuple(xml_tuple[0])[0].get_text()
        self.assertEqual(text, ' 195ISSN 1407 - 0618 \n \n')

    def test_buffer_to_xml_object(self):
        pdf_path = get_test_pdf_path()
        with open(pdf_path, 'rb') as pdf:
            xml_tuple = pdf_to_xml_object(pdf)
        text = tuple(xml_tuple[0])[0].get_text()
        self.assertEqual(text, ' 195ISSN 1407 - 0618 \n \n')


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
