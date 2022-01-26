import unittest
import io
import sys

from ntm_classifier.training.image_dataset import ImageDataset
from ntm_classifier.training.trainer import Trainer


class TestTrainer(unittest.TestCase):

    def test_run(self):
        suppress_std = io.StringIO()
        sys.stdout = suppress_std
        dataset = ImageDataset(limit_n=15)
        trainer = Trainer(dataset)
        trainer.train()
        # trainer.save()
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover
