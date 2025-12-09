import unittest
from unittest.mock import MagicMock, patch
import os
from tandon_ai_doc_intel.pipeline import DocumentPipeline
from tandon_ai_doc_intel.models import DocumentResult

class TestDocumentPipeline(unittest.TestCase):

    def setUp(self):
        # Mock dependencies
        self.mock_openai_key = "sk-test"
        self.pipeline = DocumentPipeline(openai_api_key=self.mock_openai_key)
        
        # Mock internal components to avoid external calls
        self.pipeline.enricher = MagicMock()
        self.pipeline.embedding_provider = MagicMock()
        self.pipeline.vector_store = MagicMock()
        self.pipeline.digital_extractor = MagicMock()
        self.pipeline.scanned_extractor = MagicMock()

    def test_process_flow(self):
        # Setup mocks
        mock_bytes = b"%PDF-1.4..."
        
        # Mock ingestion (static method)
        with patch("tandon_ai_doc_intel.ingestion.DocumentIngestor.ingest", return_value=mock_bytes):
            # Mock classification
            with patch("tandon_ai_doc_intel.classification.DocumentClassifier.is_digital_pdf", return_value=True):
                
                # Mock extraction
                self.pipeline.digital_extractor.extract.return_value = ("Sample text", [])
                
                # Mock enrichment
                self.pipeline.enricher.chunk_text.return_value = ["Sample", "text"]
                self.pipeline.enricher.summarize.return_value = "Summary"
                self.pipeline.enricher.extract_entities.return_value = []
                self.pipeline.enricher.analyze_risk.return_value = {}
                
                # Mock embedding
                self.pipeline.embedding_provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

                # Run process
                result = self.pipeline.process("dummy.pdf")

                # Assertions
                self.assertIsInstance(result, DocumentResult)
                self.assertEqual(result.text, "Sample text")
                self.assertEqual(result.summary, "Summary")
                self.assertTrue(result.metadata["is_digital_pdf"])
                
                # Check calls
                self.pipeline.digital_extractor.extract.assert_called_once()
                self.pipeline.enricher.summarize.assert_called_once()
                self.pipeline.vector_store.add_documents.assert_called_once()

if __name__ == "__main__":
    unittest.main()

