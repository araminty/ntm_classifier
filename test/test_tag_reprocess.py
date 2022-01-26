import unittest
from numpy import testing as nptesting

from ntm_classifier.load_resources import load_classification_table
from ntm_classifier.training.tag_reprocess_util import (
    build_tag_mappings_from_group)


class TestTagReprocess(unittest.TestCase):

    def test_rebuild_primary(self):
        tags_df = load_classification_table('tags.csv')
        target = load_classification_table('primary_tags.csv')
        target = target.fillna('None')
        result = build_tag_mappings_from_group(tags_df, 'primary', store=False)
        result = result.fillna('None')

        nptesting.assert_array_equal(target.values, result.values)


if __name__ == '__main__':
    unittest.main(verbosity=2)  # pragma: no cover
