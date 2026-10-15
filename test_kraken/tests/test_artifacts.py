import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kraken_pipeline.artifacts import run_manifest
from kraken_pipeline.config import PipelineConfig, RAW


class ManifestTests(unittest.TestCase):
    def test_duplicate_distribution_metadata_uses_effective_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'segmenter'
            model.write_bytes(b'model')
            config = PipelineConfig(segmentation_model=model, reference_dir=root)
            distributions = [
                SimpleNamespace(metadata={'Name': 'numpy'}, version='2.2.6'),
                SimpleNamespace(metadata={'Name': 'numpy'}, version='1.26.4'),
            ]
            with patch('kraken_pipeline.artifacts.metadata.distributions', return_value=distributions), \
                 patch('kraken_pipeline.artifacts.metadata.version', return_value='2.2.6'):
                manifest = run_manifest(config, (RAW,), [])
            self.assertEqual(manifest['packages']['numpy'], '2.2.6')


if __name__ == '__main__':
    unittest.main()
