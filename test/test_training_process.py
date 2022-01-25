import unittest

from ntm_classifier.model_training import (
    ready_dataframe,
    CustomDataset
)
import torch
from PIL.Image import Image

from ntm_classifier.load_resources import process_mappings_group
primary_mappings = process_mappings_group('primary')


class TestPrepareTrain(unittest.TestCase):
    dataframe = ready_dataframe()
    custom_dataset = CustomDataset(dataframe)

    def test_ready_dataframe(self):
        sample_iamge = self.dataframe.loc[0, 'tensor']
        sample_label = self.dataframe.loc[0, 'primary']
        self.assertIn(sample_label, primary_mappings)
        self.assertIsInstance(sample_iamge, torch.Tensor)
        self.assertEqual(tuple(tt.shape), (1, 3, 224, 224))

    def test_ready_custom_dataset(self):
        x1, y1 = self.custom_dataset[0]
        self.assertIsInstance(y1, torch.Tensor)
        self.assertISInstance(x1, torch.Tensor)
        self.assertEqual(tuple(x1.shape), (1, 3, 224, 224))
        # self.assertISInstance(y, )
        # print(x1, y1)


# class TestTraining(unittest.TestCase):
    # def train_training(self):
        # pass

        
if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover
