"""Verify summaries reach Elasticsearch without downloading models or calling Ollama."""
import importlib
import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict('sys.modules', {'torch': MagicMock(Tensor=type('Tensor', (), {})), 'sentence_transformers': MagicMock()}):
            cls.summarizer = importlib.import_module('app.ingestion.clustering.cluster_summarizer')
            cls.builder = importlib.import_module('app.ingestion.clustering.cluster_builder')
            cls.subclusters = importlib.import_module('app.ingestion.clustering.kmeans_subclusters')
            cls.fetcher = importlib.import_module('app.ingestion.clustering.chunk_fetcher')

    def test_root_summary_is_written(self):
        writer = MagicMock(active_index='test-clusters')
        with patch.object(self.builder, 'cluster_tag_fields', return_value={'tags': ['fees']}), \
             patch.object(self.builder, 'load_all_vectors_and_ids', return_value=(np.ones((2, 384)), ['a', 'b'])), \
             patch.object(self.builder, 'run_agglomerative_clustering', return_value={'root': [0, 1]}), \
             patch.object(self.builder, 'fetch_chunk_texts', return_value=['Bank fees']) as fetch, \
             patch.object(self.builder, 'summarize_cluster', return_value=('Explains bank fees', [0.1]*384)):
            self.builder.build_all_clusters(writer)
        fetch.assert_called_once_with(['a', 'b'])
        self.assertEqual(writer.write_cluster_doc.call_args.kwargs['body']['summary'], 'Explains bank fees')

    def test_subcluster_summary_is_written(self):
        writer = MagicMock(active_index='test-clusters')
        doc = {'cluster_id': 'sub', 'chunk_ids': ['a']}
        with patch.object(self.subclusters, 'cluster_tag_fields', return_value={'tags': ['atm']}), \
             patch.object(self.subclusters, 'fetch_chunk_texts', return_value=['ATM limits']), \
             patch.object(self.subclusters, 'summarize_cluster', return_value=('Explains ATM limits', [0.1]*384)):
            self.subclusters.write_subclusters_to_es(writer, 'root', [doc])
        self.assertEqual(writer.write_cluster_doc.call_args.kwargs['body']['summary'], 'Explains ATM limits')

    def test_fetch_uses_keyword_field_directly(self):
        with patch.object(self.fetcher, 'get_es_connection') as connection:
            connection.return_value.search.return_value = {'hits': {'hits': [{'_source': {'content': 'text'}}]}}
            self.assertEqual(self.fetcher.fetch_chunk_texts(['a']), ['text'])
            query = connection.return_value.search.call_args.kwargs['body']['query']
            self.assertEqual(query, {'terms': {'chunk_id': ['a']}})

    def test_missing_model_error_includes_ollama_explanation(self):
        with patch.object(self.summarizer.requests, 'post') as post:
            response = post.return_value
            response.status_code = 404
            response.json.return_value = {'error': 'model gemma3:12b not found'}
            response.raise_for_status.side_effect = self.summarizer.requests.HTTPError('404')
            with self.assertRaisesRegex(RuntimeError, 'model gemma3:12b not found'):
                self.summarizer.summarize_cluster(['Bank fees'])

    def test_summary_failures_do_not_become_placeholder_text(self):
        with patch.object(self.summarizer.requests, 'post') as post:
            with self.assertRaises(ValueError):
                self.summarizer.summarize_cluster([])
            post.assert_not_called()
            post.return_value.json.return_value = {'response': ' '}
            with self.assertRaisesRegex(RuntimeError, 'empty summary'):
                self.summarizer.summarize_cluster(['Bank fees'])
            post.side_effect = ConnectionError('unreachable')
            with self.assertRaisesRegex(RuntimeError, 'unreachable'):
                self.summarizer.summarize_cluster(['Bank fees'])


if __name__ == '__main__':
    unittest.main()
