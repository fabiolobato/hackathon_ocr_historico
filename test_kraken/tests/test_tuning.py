import hashlib
import json
import tempfile
import unittest

import pytest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kraken_pipeline.config import KRAKEN_BASELINE, PipelineConfig, PreprocessingConfig, ScoringConfig
from kraken_pipeline.tuning import (
    Parameter, SearchConfig, SearchSpace, choose_best,
    evaluate_heldout, generate_trials, load_tuned_config, run_search, split_pages,
)


class SearchTests(unittest.TestCase):
    def test_grid_and_seeded_random_cover_distinct_candidates(self) -> None:
        base = PreprocessingConfig()
        grid = generate_trials(base, SearchConfig(method='grid'))
        self.assertEqual(len(grid), 26)
        first = generate_trials(base, SearchConfig(seed=42))
        self.assertEqual(first, generate_trials(base, SearchConfig(seed=42)))
        self.assertNotEqual(first, generate_trials(base, SearchConfig(seed=43)))
        self.assertEqual(len({trial.variant.name for trial in first}), 12)
        self.assertTrue(all(trial.variant.segmenter == 'neural' for trial in grid))
        self.assertTrue(all(trial.variant.operations[-1] == 'kraken_nlbin' for trial in grid))
        self.assertEqual(base.denoise_strength, 7)

    def test_grid_budget_and_duplicate_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'budget'):
            generate_trials(PreprocessingConfig(), SearchConfig(method='grid', max_trials=2))
        space = SearchSpace('denoise', ('denoise',), (Parameter('denoise_strength', (3, 3)),))
        with self.assertRaisesRegex(ValueError, 'unique'):
            generate_trials(PreprocessingConfig(), SearchConfig(spaces=(space,)))

    def test_page_split_is_reproducible_and_disjoint(self) -> None:
        pages = [Path(f'{n}.jpg') for n in range(12)]
        tune, heldout = split_pages(pages)
        self.assertEqual((len(tune), len(heldout)), (9, 3))
        self.assertFalse(set(tune) & set(heldout))
        self.assertEqual((tune, heldout), split_pages(pages))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            split_pages([Path('a.jpg'), Path('a.jpg')], 1)

    def test_selection_excludes_failures_and_keeps_baseline_on_ties(self) -> None:
        baseline = {'wer': 0.4, 'complete': True}
        failed = {'wer': 0.01, 'complete': False}
        self.assertIs(choose_best([failed, {'wer': 0.4, 'complete': True}], baseline), baseline)
        better = {'wer': 0.3, 'complete': True}
        self.assertIs(choose_best([failed, better], baseline), better)
        with self.assertRaisesRegex(ValueError, 'complete WER'):
            choose_best([better], {'wer': None, 'complete': False})

    def test_config_handoff_validates_model_and_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'weights'
            weights.write_bytes(b'model')
            config = PipelineConfig(recognition_model=weights)
            trial = generate_trials(config.preprocessing, SearchConfig())[0]
            best = root / 'best.json'
            best.write_text(json.dumps({'selected': asdict(trial), 'scoring': asdict(config.scoring),
                                        'recognition_model_sha256': hashlib.sha256(b'model').hexdigest()}))
            tuned, variant = load_tuned_config(config, best)
            self.assertEqual(tuned.preprocessing, trial.preprocessing)
            self.assertEqual(variant.operations, trial.variant.operations)
            with self.assertRaisesRegex(ValueError, 'scoring'):
                load_tuned_config(PipelineConfig(recognition_model=weights, scoring=ScoringConfig(True)), best)
            weights.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'weights differ'):
                load_tuned_config(config, best)

    def test_search_saves_baseline_candidate_provenance_and_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'weights'
            weights.write_bytes(b'model')
            config = PipelineConfig(output_dir=root, recognition_model=weights)
            spaces = (SearchSpace('denoise', ('denoise',), (Parameter('denoise_strength', (3, 12)),)),)
            runs = []

            def execute(config: PipelineConfig, engine: object, variants: tuple,
                        images: list[Path], fail_fast: bool = False) -> Path:
                run = config.output_dir / f'run_{len(runs)}'
                run.mkdir()
                runs.append(run)
                rows = [{'variant': variant.name, 'wer': 0.4 if variant.name == KRAKEN_BASELINE.name
                         else (0.3 if config.preprocessing.denoise_strength == 3 else 0.8),
                         'complete': True, 'reference_status': 'provisional'} for variant in variants]
                (run / 'summary.json').write_text(json.dumps(rows))
                (run / 'config.json').write_text(json.dumps({key: {} for key in (
                    'model_sha256', 'image_sha256', 'reference_sha256', 'pipeline_sha256')}))
                return run

            artifacts = SimpleNamespace(file_hash=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
                                        write_json=lambda p, value: p.write_text(json.dumps(value, default=str)))
            experiments = SimpleNamespace(run_experiments=execute)
            with patch.dict('sys.modules', {'kraken_pipeline.artifacts': artifacts,
                                            'kraken_pipeline.experiments': experiments}):
                result = run_search(config, object(), [Path('1.jpg')], SearchConfig(method='grid', spaces=spaces))
                selected = json.loads((result / 'best.json').read_text())
                self.assertTrue(selected['improved'])
                self.assertEqual(selected['tuning_wer'], 0.3)
                self.assertEqual(selected['baseline_variant'], 'kraken_neural')
                self.assertEqual(load_tuned_config(config, result / 'best.json')[0].preprocessing.denoise_strength, 3)
                controls = json.loads((runs[0] / 'summary.json').read_text())
                self.assertEqual({row['variant'] for row in controls}, {'raw', 'kraken_neural'})
                selection = json.loads((result / 'best.json').read_text())
                selection['selected']['variant']['segmenter'] = 'legacy'
                old_profile = root / 'old_profile.json'
                old_profile.write_text(json.dumps(selection))
                with self.assertRaisesRegex(ValueError, 'neural'):
                    load_tuned_config(config, old_profile)
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    evaluate_heldout(config, object(), result / 'best.json', [Path('1.jpg')])


if __name__ == '__main__':
    unittest.main()


def test_select_pipeline_rejects_legacy_profiles(tmp_path: Path) -> None:
    from kraken_pipeline.tuning import select_search_result

    paths = []
    for name, wer in [('first', 0.7), ('second', 0.6)]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / 'best.json').write_text(json.dumps({
            'scoring': {}, 'recognition_model_sha256': 'same-model',
            'tuning_pages': ['1.jpg'], 'baseline_variant': 'kraken_neural',
            'controls_run': 'same-controls', 'tuning_wer': wer,
            'selected': {'variant': {'segmenter': 'neural'}},
        }))
        paths.append(directory)
    with patch('kraken_pipeline.tuning._verify_comparison'):
        assert select_search_result(tuple(paths)) == paths[1] / 'best.json'
        old = json.loads((paths[1] / 'best.json').read_text())
        old['selected']['variant']['segmenter'] = 'legacy'
        (paths[1] / 'best.json').write_text(json.dumps(old))
        with pytest.raises(ValueError, match='neural'):
            select_search_result(tuple(paths))
