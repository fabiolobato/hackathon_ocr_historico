"""Reproducible discrete searches; OCR execution remains in experiments.py."""
import json
import random
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import product
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .config import NEURAL_BASELINE, RAW, PipelineConfig, PreprocessingConfig, Variant

if TYPE_CHECKING:
    from .engine import KrakenEngine


@dataclass(frozen=True)
class Parameter:
    name: str
    values: tuple[float | int, ...]


@dataclass(frozen=True)
class SearchSpace:
    name: str
    operations: tuple[str, ...]
    parameters: tuple[Parameter, ...]


SINGLE_SPACES = (
    SearchSpace('denoise', ('denoise',), (Parameter('denoise_strength', (3, 7, 12)),)),
    SearchSpace('contrast', ('contrast',), (Parameter('contrast_factor', (1.2, 1.5, 1.8)),)),
    SearchSpace('clahe', ('clahe',), (Parameter('clahe_clip', (1.5, 2.0, 3.0)), Parameter('clahe_grid', (4, 8)))),
    SearchSpace('illumination', ('illumination',), (Parameter('background_sigma', (15, 25, 40)),)),
    SearchSpace('upscale', ('upscale',), (Parameter('upscale_factor', (1.5, 2.0)),)),
    SearchSpace('deskew', ('deskew',), (Parameter('deskew_limit', (2, 5, 8)),)),
    SearchSpace('dewarp', ('dewarp',), (Parameter('dewarp_limit', (10, 25, 40)),)),
    SearchSpace('nlbin', (), (Parameter('nlbin_threshold', (0.4, 0.5, 0.6)),)),
)
COMBINATION_SPACES = (
    SearchSpace('denoise_clahe', ('denoise', 'clahe'), (
        Parameter('denoise_strength', (3, 7, 12)), Parameter('clahe_clip', (1.5, 2.0, 3.0)),
        Parameter('clahe_grid', (4, 8)))),
    SearchSpace('illumination_clahe', ('illumination', 'clahe'), (
        Parameter('background_sigma', (15, 25, 40)), Parameter('clahe_clip', (1.5, 2.0, 3.0)))),
)

BINARIZATION_SPACES = (
    SearchSpace('limiar_nlbin', (), (Parameter('nlbin_threshold', (0.2, 0.35, 0.5, 0.65, 0.8)),)),
)
REFINEMENT_SPACES = (
    SearchSpace('contraste_limiar', ('contrast',), (
        Parameter('contrast_factor', (1.2, 1.8)), Parameter('nlbin_threshold', (0.4, 0.6)))),
    SearchSpace('denoise_limiar', ('denoise',), (
        Parameter('denoise_strength', (3, 12)), Parameter('nlbin_threshold', (0.4, 0.6)))),
    SearchSpace('ampliacao_limiar', ('upscale',), (
        Parameter('upscale_factor', (1.5, 2.0)), Parameter('nlbin_threshold', (0.4, 0.6)))),
)
@dataclass(frozen=True)
class SearchConfig:
    method: Literal['random', 'grid'] = 'random'
    n_trials: int = 12
    max_trials: int = 64
    seed: int = 42
    spaces: tuple[SearchSpace, ...] = SINGLE_SPACES
    segmenter: Literal['neural'] = 'neural'
    include_raw: bool = True


@dataclass(frozen=True)
class Trial:
    variant: Variant
    preprocessing: PreprocessingConfig


def generate_trials(base: PreprocessingConfig, search: SearchConfig) -> tuple[Trial, ...]:
    if search.segmenter != 'neural':
        raise ValueError('Este pipeline usa somente segmenter=neural.')
    if search.method not in ('random', 'grid') or min(search.n_trials, search.max_trials) < 1:
        raise ValueError('Choose random/grid and positive trial budgets.')
    total = sum(prod(len(p.values) for p in space.parameters) for space in search.spaces)
    if search.method == 'grid' and total > search.max_trials:
        raise ValueError(f'Grid contains {total} trials; budget is {search.max_trials}. Narrow the grid or increase the budget.')
    if total > 100_000:
        raise ValueError('Discrete search space exceeds 100000 candidates; narrow parameter ranges.')
    candidates = []
    names = set()
    for space in search.spaces:
        keys = [parameter.name for parameter in space.parameters]
        if space.name in names or len(keys) != len(set(keys)):
            raise ValueError('Search-space names and parameter names must be unique.')
        names.add(space.name)
        if any(not parameter.values or len(set(parameter.values)) != len(parameter.values)
               for parameter in space.parameters):
            raise ValueError('Parameter values must be nonempty and unique.')
        for values in product(*(parameter.values for parameter in space.parameters)):
            preprocessing = replace(base, **dict(zip(keys, values)))
            variant = Variant(f'trial_{len(candidates):04d}_{space.name}',
                              space.operations + ('kraken_nlbin',), search.segmenter)
            candidates.append(Trial(variant, preprocessing))
    if not candidates:
        raise ValueError('The search space is empty.')
    if search.method == 'grid':
        if len(candidates) > search.max_trials:
            raise ValueError(f'Grid contains {len(candidates)} trials; budget is {search.max_trials}. Narrow the grid or increase the budget.')
        return tuple(candidates)
    count = min(search.n_trials, len(candidates))
    if count > search.max_trials:
        raise ValueError('Random-search trial count exceeds max_trials.')
    return tuple(random.Random(search.seed).sample(candidates, count))


def split_pages(images: list[Path], heldout_count: int = 3,
                seed: int = 42) -> tuple[list[Path], list[Path]]:
    if not 0 < heldout_count < len(images):
        raise ValueError('Reserve at least one page and keep at least one tuning page.')
    if len({path.resolve() for path in images}) != len(images):
        raise ValueError('Duplicate pages cannot be split independently.')
    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    return shuffled[heldout_count:], shuffled[:heldout_count]


def choose_best(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    if baseline.get('wer') is None or not baseline.get('complete'):
        raise ValueError('The neural baseline must have a complete WER before selecting parameters.')
    valid = [row for row in rows if row.get('complete') and row.get('wer') is not None]
    return min([baseline] + valid, key=lambda row: row['wer'])


def _read_summary(run_dir: Path) -> list[dict[str, Any]]:
    return json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))


def _verify_comparison(control: Path, candidate: Path) -> None:
    reference = json.loads((control / 'config.json').read_text(encoding='utf-8'))
    trial = json.loads((candidate / 'config.json').read_text(encoding='utf-8'))
    for key in ('model_sha256', 'image_sha256', 'reference_sha256', 'pipeline_sha256'):
        if reference[key] != trial[key]:
            raise ValueError(f'{key} changed during tuning; do not rank incompatible runs.')


def _save_leaderboard(directory: Path, rows: list[dict[str, Any]]) -> None:
    from .artifacts import write_json

    write_json(directory / 'summary.json', rows)
    columns = ('variant', 'wer', 'delta_wer', 'complete', 'nonempty_pages', 'empty_pages', 'reference_status', 'run_dir')
    with (directory / 'leaderboard.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def run_search(config: PipelineConfig, engine: 'KrakenEngine', images: list[Path],
               search: SearchConfig = SearchConfig()) -> Path:
    from .artifacts import file_hash, write_json
    from .experiments import run_experiments

    trials = generate_trials(config.preprocessing, search)
    directory = config.output_dir / ('tuning_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    directory.mkdir(parents=True)
    run_config = replace(config, output_dir=directory)
    write_json(directory / 'search.json', {'search': asdict(search), 'pages': [str(p) for p in images],
                                         'trials': [asdict(trial) for trial in trials]})
    controls = run_experiments(run_config, engine, (RAW, NEURAL_BASELINE), images, fail_fast=True)
    control = NEURAL_BASELINE
    baseline = {**next(row for row in _read_summary(controls) if row['variant'] == control.name),
                'baseline_variant': control.name, 'delta_wer': 0.0}
    alternatives = [row for row in _read_summary(controls) if row['variant'] == RAW.name]
    alternatives = alternatives if search.include_raw else []
    choose_best([], baseline)
    rows = []
    for trial in trials:
        trial_config = replace(run_config, preprocessing=trial.preprocessing)
        run = run_experiments(trial_config, engine, (trial.variant,), images)
        _verify_comparison(controls, run)
        row = {**_read_summary(run)[0], 'run_dir': str(run), 'trial': asdict(trial)}
        row['baseline_variant'] = control.name
        row['delta_wer'] = row['wer'] - baseline['wer'] if row['wer'] is not None and baseline['wer'] is not None else None
        rows.append(row)
        write_json(directory / 'trials.json', rows)
        _save_leaderboard(directory, [baseline] + alternatives + rows)
    best = choose_best(alternatives + rows, baseline)
    fallback = RAW if best['variant'] == RAW.name else control
    selected = best.get('trial', {'variant': asdict(fallback), 'preprocessing': asdict(PreprocessingConfig())})
    selection = {'selected': selected, 'tuning_wer': best['wer'], 'baseline_wer': baseline['wer'],
                 'improved': best['wer'] < baseline['wer'], 'controls_run': str(controls),
                 'baseline_variant': control.name,
                 'reference_status': best['reference_status'], 'scoring': asdict(config.scoring),
                 'recognition_model_sha256': file_hash(config.recognition_model),
                 'tuning_pages': [str(p) for p in images]}
    write_json(directory / 'best.json', selection)
    return directory


def select_search_result(search_dirs: tuple[Path, ...]) -> Path:
    """Escolhe entre buscas comparáveis usando somente o WER de ajuste."""
    if not search_dirs:
        raise ValueError('Informe pelo menos uma busca concluída.')
    paths = [directory / 'best.json' for directory in search_dirs]
    selections = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    for selection in selections:
        if selection['selected']['variant']['segmenter'] != 'neural':
            raise ValueError('Perfil incompatível: refaça a busca com segmentação neural.')
    first = selections[0]
    for selection in selections[1:]:
        keys = ('scoring', 'recognition_model_sha256', 'tuning_pages', 'baseline_variant')
        for key in keys:
            if first[key] != selection[key]:
                raise ValueError(f'Buscas incompatíveis: {key}.')
        _verify_comparison(Path(first['controls_run']), Path(selection['controls_run']))
    return min(zip(paths, selections), key=lambda item: item[1]['tuning_wer'])[0]


def load_tuned_config(config: PipelineConfig, best_path: Path) -> tuple[PipelineConfig, Variant]:
    import hashlib

    result = json.loads(best_path.read_text(encoding='utf-8'))
    if asdict(config.scoring) != result['scoring']:
        raise ValueError('Use the same scoring policy as the tuning run.')
    if config.recognition_model is None:
        raise ValueError('Select the recognition model before importing tuning results.')
    digest = hashlib.sha256()
    with config.recognition_model.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != result['recognition_model_sha256']:
        raise ValueError('Recognition weights differ from the tuning run.')
    selected = result['selected']
    preprocessing = PreprocessingConfig(**selected['preprocessing'])
    variant = selected['variant']
    return replace(config, preprocessing=preprocessing), Variant(
        'tuned_' + variant['name'], tuple(variant['operations']), variant['segmenter'])


def evaluate_heldout(config: PipelineConfig, engine: 'KrakenEngine', best_path: Path,
                     images: list[Path]) -> Path:
    from .experiments import run_experiments

    selection = json.loads(best_path.read_text(encoding='utf-8'))
    if {Path(path).name for path in selection['tuning_pages']} & {path.name for path in images}:
        raise ValueError('Held-out pages overlap with pages used for tuning.')
    tuned_config, tuned_variant = load_tuned_config(config, best_path)
    # Keep native nlbin defaults fixed for the control, even if nlbin was tuned.
    control_variant = NEURAL_BASELINE
    control = run_experiments(config, engine, (control_variant,), images, fail_fast=True)
    tuned = run_experiments(tuned_config, engine, (tuned_variant,), images, fail_fast=True)
    from .artifacts import write_json

    baseline = _read_summary(control)[0]
    row = _read_summary(tuned)[0]
    row['baseline_variant'] = control_variant.name
    row['delta_wer'] = row['wer'] - baseline['wer'] if row['wer'] is not None and baseline['wer'] is not None else None
    write_json(tuned / 'summary.json', [baseline, row])
    write_json(tuned / 'heldout.json', {'selection': str(best_path), 'controls_run': str(control),
                                      'pages': [str(p) for p in images]})
    return tuned
