import unittest

from ntm_classifier.training.image_dataset import ImageDataset

import torch

from ntm_classifier.load_resources import process_mappings_group
primary_mappings = process_mappings_group('primary')
lowers = [label.lower() for label in primary_mappings.values()]


class TestDataset(unittest.TestCase):
    dataset = ImageDataset(limit_n=5)

    def test_correct_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        i_device = self.dataset[:]['image'].device.type
        r_device = self.dataset[:]['result'].device.type
        self.assertEqual(i_device, device)
        self.assertEqual(r_device, device)

    def test_single_shape(self):
        sample = self.dataset[0]
        image = sample['image']
        shape = image.shape
        self.assertTrue(shape[0], 1)  # singleton batch
        self.assertTrue(shape[1], 3)  # RGB channels

    def test_slice_shape(self):
        sample = self.dataset[0:4]
        image = sample['image']
        shape = image.shape
        self.assertTrue(shape[0], 4)  # batch of four
        self.assertTrue(shape[1], 3)  # RGB channels

    def assert_output_format(self):
        ytype = self.dataset[0]['result'].dtype
        self.assertTrue(ytype == torch.float)

    def test_image_resolution(self):
        shape = self.dataset[0:5]['image'].shape
        self.assertEqual(tuple(shape)[2:], (224, 224))

    def test_limits(self):
        sample = self.dataset[:5]
        image = sample['image']

        self.assertTrue(image.max().item() > 2)
        self.assertTrue(image.min().item() < 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover
