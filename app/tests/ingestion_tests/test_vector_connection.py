import unittest
from unittest.mock import patch

from app.ingestion.clustering import vector_loader


class VectorConnectionTests(unittest.TestCase):
    def test_loader_uses_configured_endpoint_and_reuses_client(self):
        with patch.object(vector_loader, 'es_connection', None), \
             patch.object(vector_loader, 'ES_HOST', 'http://elasticsearch:9200'), \
             patch.object(vector_loader, 'Elasticsearch') as client, \
             patch.dict('os.environ', {}, clear=True):
            first = vector_loader.VectorLoader(index='es_documents')
            second = vector_loader.get_es_connection()
            client.assert_called_once_with('http://elasticsearch:9200')
            self.assertIs(first.es, second)

    def test_connection_supports_configured_credentials(self):
        with patch.object(vector_loader, 'Elasticsearch') as client, \
             patch.dict('os.environ', {'ES_USERNAME': 'test-user', 'ES_PASSWORD': 'test-password'}):
            vector_loader.es_connect('http://elasticsearch:9200')
            client.assert_called_once_with('http://elasticsearch:9200',
                                           basic_auth=('test-user', 'test-password'))


if __name__ == '__main__':
    unittest.main()
