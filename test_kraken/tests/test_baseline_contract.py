"""Check native API dispatch without installing scientific dependencies locally."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock, patch

from kraken_pipeline.config import KRAKEN_BASELINE, NEURAL_BASELINE, RAW, PipelineConfig, PreprocessingConfig


def source_function(path: str, name: str, namespace: dict[str, Any]) -> Callable[..., Any]:
    tree = ast.parse(Path(path).read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(module, path, 'exec'), namespace)
    return namespace[name]


class BaselineContractTests(unittest.TestCase):
    def test_native_binarizer_dispatch(self) -> None:
        image, binarized = object(), object()
        nlbin = MagicMock(return_value=binarized)
        function = source_function('kraken_pipeline/preprocessing.py', 'process_image', {})
        with patch.dict('sys.modules', {'kraken': SimpleNamespace(),
                                        'kraken.binarization': SimpleNamespace(nlbin=nlbin)}):
            self.assertIs(function(image, 'kraken_nlbin', PreprocessingConfig()), binarized)
        nlbin.assert_called_once_with(image, threshold=0.5)

    def test_native_neural_only_dispatch(self) -> None:
        self.assertEqual(KRAKEN_BASELINE.operations, ('kraken_nlbin',))
        self.assertIs(KRAKEN_BASELINE, NEURAL_BASELINE)
        config = object()
        predict = MagicMock(return_value='baselines')
        engine = SimpleNamespace(seg_config=config, segmenter=SimpleNamespace(predict=predict))
        function = source_function('kraken_pipeline/engine.py', 'segment', {})
        self.assertEqual(function(engine, 'image'), 'baselines')
        predict.assert_called_once_with('image', config)

    def test_fixed_baseline_ignores_tuned_threshold_and_raw_is_unmodified(self) -> None:
        image = MagicMock()
        copy = image.copy.return_value
        process = MagicMock(return_value='binary')
        namespace = {'PreprocessingConfig': PreprocessingConfig, 'process_image': process}
        function = source_function('kraken_pipeline/experiments.py', 'prepare_image', namespace)
        config = PipelineConfig(preprocessing=PreprocessingConfig(nlbin_threshold=0.9))
        prepared, details = function(image, KRAKEN_BASELINE, None, config)
        self.assertEqual(prepared, 'binary')
        self.assertEqual(details[0]['threshold'], 0.5)
        self.assertEqual(process.call_args.args[2].nlbin_threshold, 0.5)
        process.reset_mock()
        self.assertEqual(function(image, RAW, None, config), (copy, []))
        process.assert_not_called()

    def test_comparison_delta_uses_neural_baseline(self) -> None:
        function = source_function('kraken_pipeline/experiments.py', 'summarize', {})
        rows = [{'variant': name, 'wer': errors / 10, 'status': 'ok', 'errors': errors,
                 'reference_words': 10, 'reference_status': 'reviewed'}
                for name, errors in [('raw', 8), ('kraken_neural', 5), ('trial', 3)]]
        result = function(rows, 1)
        self.assertEqual(result[-1]['baseline_variant'], 'kraken_neural')
        self.assertAlmostEqual(result[-1]['delta_wer'], -0.2)


if __name__ == '__main__':
    unittest.main()
