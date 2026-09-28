import unittest
from unittest.mock import MagicMock, patch
from app.ingestion.clustering import tag_enrichment as tags


class TagTests(unittest.TestCase):
    def test_tags_are_normalized_and_invalid_output_rejected(self):
        self.assertEqual(tags.normalize_tags(['ATM Fees', ' atm   fees ', 'Withdrawal']), ['atm fees', 'withdrawal'])
        for value in ([], 'atm', [42]):
            with self.assertRaises(ValueError):
                tags.normalize_tags(value)

    def test_chunk_backfill_preserves_metadata_and_skips_completed_chunks(self):
        es = MagicMock()
        hits = [
            {'_id': 'a', '_source': {'content': 'fees', 'chunk_metadata': {'page': 2}}},
            {'_id': 'b', '_source': {'chunk_metadata': {'tags': ['atm']}, 'tags_vector': [0.1]*384}},
        ]
        with patch.object(tags, 'scan', return_value=hits), \
             patch.object(tags, 'generate_tags', return_value=['fees']), \
             patch.object(tags, 'embed_tags', return_value=[0.1]*384):
            self.assertEqual(tags.enrich_chunks(es), 1)
        doc = es.update.call_args.kwargs['doc']
        self.assertEqual(doc['chunk_metadata'], {'page': 2, 'tags': ['fees']})
        self.assertEqual(len(doc['tags_vector']), 384)
        self.assertNotIn('content', doc)
        self.assertNotIn('doc_as_upsert', es.update.call_args.kwargs)

    def test_cluster_aggregates_member_tags(self):
        es = MagicMock()
        es.mget.return_value = {'docs': [
            {'found': True, '_source': {'chunk_metadata': {'tags': ['fees', 'atm']}}},
            {'found': True, '_source': {'chunk_metadata': {'tags': ['fees', 'cards']}}},
        ]}
        with patch.object(tags, 'embed_tags', return_value=[0.1]*384):
            fields = tags.cluster_tag_fields(['a', 'b'], es)
        self.assertEqual(fields['tags'], ['fees', 'atm', 'cards'])

    def test_generation_failure_does_not_write_chunk(self):
        es = MagicMock()
        with patch.object(tags, 'scan', return_value=[{'_id': 'a', '_source': {'content': 'fees'}}]), \
             patch.object(tags, 'generate_tags', side_effect=RuntimeError('model unavailable')):
            with self.assertRaises(RuntimeError):
                tags.enrich_chunks(es)
        es.update.assert_not_called()

    def test_backfill_updates_both_cluster_levels_by_concrete_index(self):
        es = MagicMock()
        with patch.object(tags, 'get_es_connection', return_value=es), \
             patch.object(tags, 'enrich_chunks', return_value=2), \
             patch.object(tags, 'scan', return_value=[
                 {'_index': 'clusters-current', '_id': 'root', '_source': {'chunk_ids': ['a', 'b']}},
                 {'_index': 'clusters-current', '_id': 'sub', '_source': {'chunk_ids': ['a']}},
             ]), patch.object(tags, 'cluster_tag_fields', return_value={'tags': ['fees'], 'tags_vector': [0.1]*384}):
            self.assertEqual(tags.backfill_tags(), {'chunks_updated': 2, 'clusters_updated': 2})
        self.assertEqual([call.kwargs['id'] for call in es.update.call_args_list], ['root', 'sub'])
        self.assertTrue(all(call.kwargs['index'] == 'clusters-current' for call in es.update.call_args_list))

    def test_model_failure_retries_with_structured_output(self):
        failed = MagicMock(ok=False, status_code=500, text='token repeat limit')
        success = MagicMock(ok=True)
        success.json.return_value = {'response': '{"tags": ["ATM Fees"]}'}
        with patch.object(tags.requests, 'post', side_effect=[failed, success]) as post:
            self.assertEqual(tags.generate_tags('ATM fees apply'), ['atm fees'])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs['json']['format']['type'], 'object')

    def test_invalid_json_retries_with_shorter_excerpt(self):
        bad = MagicMock(ok=True)
        bad.json.return_value = {'response': '{"tags": ["unfinished'}
        good = MagicMock(ok=True)
        good.json.return_value = {'response': '{"tags": ["bank fees"]}'}
        with patch.object(tags.requests, 'post', side_effect=[bad, good]) as post:
            self.assertEqual(tags.generate_tags('Bank fees ' * 200), ['bank fees'])
        prompts = [call.kwargs['json']['prompt'] for call in post.call_args_list]
        self.assertLess(len(prompts[1]), len(prompts[0]))
