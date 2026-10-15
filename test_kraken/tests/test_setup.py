import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kraken_pipeline.setup import dependencies_ready, ensure_dependencies


class SetupTests(unittest.TestCase):
    def test_ready_environment_skips_installation(self) -> None:
        with patch('kraken_pipeline.setup.dependencies_ready', return_value=True), \
             patch('kraken_pipeline.setup.install_dependencies') as install, \
             patch.dict('sys.modules', {'numpy': SimpleNamespace(__version__='2.2.6')}), \
             patch('kraken_pipeline.setup.metadata.version', return_value='2.2.6'):
            # Limit loaded-module inspection to a single deterministic fixture.
            with patch.dict('sys.modules', {key: None for key in ('scipy', 'torch', 'torchvision', 'PIL', 'matplotlib')}):
                ensure_dependencies()
            install.assert_not_called()

    def test_stale_numpy_is_rejected_after_installation(self) -> None:
        with patch('kraken_pipeline.setup.dependencies_ready', return_value=False), \
             patch('kraken_pipeline.setup.install_dependencies') as install, \
             patch.dict('sys.modules', {'numpy': SimpleNamespace(__version__='2.1.3')}), \
             patch('kraken_pipeline.setup.metadata.version', return_value='2.2.6'):
            with self.assertRaisesRegex(RuntimeError, 'Reinicie a sessão do Colab'):
                ensure_dependencies()
            install.assert_called_once()

    def test_git_commit_and_versions_are_checked_without_importing_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            requirements = Path(directory) / 'requirements.txt'
            requirements.write_text('numpy==2.2.6\nkraken @ git+https://github.com/mittagessen/kraken.git@abc123\n')
            dist = SimpleNamespace(read_text=lambda name: json.dumps({'vcs_info': {'commit_id': 'abc123'}}))
            with patch('kraken_pipeline.setup.metadata.version', return_value='2.2.6'), \
                 patch('kraken_pipeline.setup.metadata.distribution', return_value=dist):
                self.assertTrue(dependencies_ready(requirements))
            with patch('kraken_pipeline.setup.metadata.version', return_value='2.1.3'):
                self.assertFalse(dependencies_ready(requirements))


if __name__ == '__main__':
    unittest.main()
