"""Offline regression tests: no model downloads or running infrastructure required."""
import asyncio
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.utils.kafka_delivery import produce_confirmed


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Replace extraction/model dependencies, retaining the actual service and workers.
        with patch.dict('sys.modules', {
            'torch': MagicMock(),
            'sentence_transformers': MagicMock(),
            'app.ingestion.minio_client': MagicMock(),
            'app.workflow.ingestion_workflow': MagicMock(),
        }):
            cls.service_module = importlib.import_module('app.services.ingestion_service')
            cls.embed_module = importlib.import_module('app.workers.embed_worker')
            cls.index_module = importlib.import_module('app.workers.indexer_worker')

    def producer(self, error=None):
        producer = MagicMock()
        callbacks = []
        producer.produce.side_effect = lambda **kw: callbacks.append(kw['on_delivery'])
        def flush(timeout=None):
            while callbacks:
                callbacks.pop(0)(error, None)
            return 0
        producer.flush.side_effect = flush
        return producer

    def chunk(self):
        return dict(chunk_id='c1', doc_id='d1', file_name='test.pdf',
                    file_type='pdf', content='sample text', embedding_vector=[0.1] * 384)

    def message(self):
        msg = MagicMock()
        msg.error.return_value = None
        msg.value.return_value = json.dumps(self.chunk()).encode()
        return msg

    def test_delivery_failure_is_not_hidden_by_empty_queue(self):
        with self.assertRaisesRegex(RuntimeError, 'broker rejected'):
            produce_confirmed(self.producer('broker rejected'), topic='chunks', key=b'c', value=b'v')

    def test_delivery_timeout(self):
        producer = MagicMock()
        producer.flush.return_value = 1
        with self.assertRaises(TimeoutError):
            produce_confirmed(producer, topic='chunks', key=b'c', value=b'v')

    def test_mapping_graph_result_publishes_chunks(self):
        service = self.service_module.IngestionService.__new__(self.service_module.IngestionService)
        service.storage = MagicMock()
        service.producer = self.producer()
        service.chunks_topic = 'custom.chunks'
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'test.pdf').touch()
            with patch.object(self.service_module.IngestionLangGraph, 'ainvoke',
                              AsyncMock(return_value={'chunks': [self.chunk()], 'error': None})):
                self.assertEqual(asyncio.run(service.bulk_ingest(folder)), 1)
        payload = json.loads(service.producer.produce.call_args.kwargs['value'])
        self.assertEqual(payload['chunk_id'], 'c1')
        self.assertEqual(service.producer.produce.call_args.kwargs['topic'], 'custom.chunks')

    def test_empty_or_failed_graph_is_not_success(self):
        service = self.service_module.IngestionService.__new__(self.service_module.IngestionService)
        service.storage = MagicMock()
        service.producer = self.producer()
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'test.pdf').touch()
            for result in ({'chunks': None}, {'error': 'extraction failed'}):
                with patch.object(self.service_module.IngestionLangGraph, 'ainvoke', AsyncMock(return_value=result)):
                    self.assertEqual(asyncio.run(service.bulk_ingest(folder)), 0)
        service.producer.produce.assert_not_called()

    def test_embedding_failure_does_not_commit_or_continue(self):
        worker = self.embed_module.EmbedWorker.__new__(self.embed_module.EmbedWorker)
        worker.consumer = MagicMock()
        worker.consumer.poll.return_value = self.message()
        worker.producer = self.producer('broker rejected')
        worker.input_topic, worker.output_topic, worker.model_name = 'chunks', 'embedded', 'fake'
        worker.get_embedding = MagicMock(return_value=[0.1] * 384)
        with self.assertRaises(RuntimeError):
            worker.run()
        worker.consumer.commit.assert_not_called()
        worker.consumer.close.assert_called_once()
        self.assertEqual(worker.consumer.poll.call_count, 1)

    def test_index_failure_does_not_commit_or_skip(self):
        worker = self.index_module.IndexerWorker.__new__(self.index_module.IndexerWorker)
        worker.consumer = MagicMock()
        worker.consumer.poll.return_value = self.message()
        worker.input_topic, worker.index_name = 'embedded', 'documents'
        worker.es = MagicMock()
        worker.es.index.side_effect = RuntimeError('mapping rejected')
        with self.assertRaisesRegex(RuntimeError, 'mapping rejected'):
            worker.run()
        worker.consumer.commit.assert_not_called()
        worker.consumer.close.assert_called_once()
        worker.es.close.assert_called_once()
        self.assertEqual(worker.consumer.poll.call_count, 1)

    def test_successful_index_commits_after_write(self):
        worker = self.index_module.IndexerWorker.__new__(self.index_module.IndexerWorker)
        worker.consumer, worker.es = MagicMock(), MagicMock()
        worker.input_topic, worker.index_name = 'embedded', 'documents'
        msg = self.message()
        worker.consumer.poll.side_effect = [msg] + [None] * 40
        events = []
        worker.es.index.side_effect = lambda **kw: events.append('index')
        worker.consumer.commit.side_effect = lambda *a, **kw: events.append('commit')
        worker.run()
        self.assertEqual(events, ['index', 'commit'])
        self.assertEqual(worker.es.index.call_args.kwargs['id'], 'c1')

    def test_batch_waits_for_embedding_before_indexing(self):
        service = self.service_module.IngestionService.__new__(self.service_module.IngestionService)
        service.bulk_ingest = AsyncMock(return_value=1)
        events = []
        service.run_embedding_phase = lambda: events.append('embed')
        service.run_indexing_phase = lambda: events.append('index')
        asyncio.run(service.execute_complete_pipeline('data'))
        self.assertEqual(events, ['embed', 'index'])


if __name__ == '__main__':
    unittest.main()
