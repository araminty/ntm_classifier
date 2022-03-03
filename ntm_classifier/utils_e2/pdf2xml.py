import os
from io import BufferedReader, BufferedWriter
from typing import Union

from pdfminer.high_level import extract_text_to_fp, extract_pages
from pdfminer.layout import LAParams
from ntm_classifier.load_resources import get_test_pdf_path

# pdf_path = '/home/airy/Documents/e2-ntm-country-parsing/latvia/1946293_LG_12_2021.pdf'
# xml_path =
# '/home/airy/Documents/e2-ntm-country-parsing/latvia/1946293_LG_12_2021.xml'


def get_xml_tmp_path(pdf_path):
    if os.path.exists('/tmp/'):
        pdf_path_end = '/'.join(pdf_path.split('/')[-3:]).\
            replace('.pdf', '.xml')
        xml_path = os.path.join('/tmp/', pdf_path_end)
        xml_dir = '/'.join(xml_path.split('/')[:-1])
        if not os.path.exists(xml_dir):
            os.makedirs(xml_dir)
        return xml_path
    else:
        return pdf_path.replace('.pdf', '.xml')


def pdf_to_xml_object(pdf: Union[BufferedReader, str]):
    if not isinstance(pdf, BufferedReader):
        pdf = str(pdf)
        with open(pdf, 'rb') as pdf_buffer:
            return pdf_to_xml_object(pdf_buffer)

    pages = extract_pages(pdf)
    return tuple(pages)


def pdf_to_xml_file(
        pdf: Union[BufferedReader, str],
        xml: Union[BufferedReader, str, None] = None,
        maxpages=20,
        word_margin=0.2,
        char_margin=1,
        line_margin=0.3):

    def buffer_to_buffer(pdf, xml):
        laparams = LAParams(
            word_margin=word_margin,
            char_margin=char_margin,
            line_margin=line_margin,
            all_texts=True,)
        extract_text_to_fp(
            pdf, xml,
            output_type='xml', maxpages=maxpages,
            layoutmode=True,
            laparams=laparams,
            output_dir='/tmp/images/',
        )

    if xml is None:
        try:
            pdf_path = str(pdf)
            xml_path = get_xml_tmp_path(pdf_path)
        except BaseException:
            if os.path.exists('/tmp/'):
                xml_path = '/tmp/xml_from_pdf.xml'
            else:
                xml_path = os.path.join(os.getcwd(), 'xml_from_pdf.xml')
        return pdf_to_xml_file(pdf, xml_path, maxpages)

    if not isinstance(pdf, BufferedReader):
        pdf = str(pdf)
        with open(pdf, 'rb') as pdf_buffer:
            return pdf_to_xml_file(pdf_buffer, xml, maxpages)

    if not isinstance(xml, BufferedWriter):
        xml = str(xml)
        with open(xml, 'wb') as xml_buffer:
            return pdf_to_xml_file(pdf, xml_buffer, maxpages)

    return buffer_to_buffer(pdf, xml)


def make_page_images(pdf_path):
    from pdf2image import convert_from_path
    if not os.path.exists(pdf_path[:-4]):
        os.mkdir(pdf_path[:-4])
    pihr = os.path.join(pdf_path[:-4], "page_images_high_res")
    if not os.path.exists(pihr):
        os.mkdir(pihr)
    if not os.path.exists('/tmp/page_images'):
        os.mkdir('/tmp/page_images')

    pages = convert_from_path(pdf_path, output_folder='/tmp/page_images/')
    for i, page in enumerate(pages):
        page.save(os.path.join(pihr,
                               f"Page-{'0'*(6-len(str(i)))}{i}.png"))
