"""Reduced search spec for a new recognizer: same stages and candidates as a full spec (so --init from that search
works), recognizer fixed in the base, recognizer-independent stages frozen at the initial choices, and the stages
that depend on what the recognizer reads re-decided (line filter, recognition image, line extraction, rescoring,
lexicon and LM choice and parameters), with a few extra candidates around the earlier optimum.

  python -m ocr_pipeline.tuning.make_reduced_spec sweeps/search_v4.json st_v3 sweeps/search_st_v3.json
"""
import argparse
import json
from pathlib import Path
from typing import Any

FROZEN = ('recognizer', 'seg_image', 'seg_params', 'text', 'decoder')  # stable across the v4 passes
EXTRA: dict[str, dict[str, Any]] = {
    'rec_image': {'channel_b': {'rec_ops': 'channel', 'params.channel': 'b'}},  # E13: best image variant for medium
    'geometry': {'upscale_25': {'geometric_ops': 'upscale', 'params.upscale_factor': 2.5}},
    'decoder_params': {f'alpha_{a}': {'set': {'post.ctc_alpha': a},
                                      'requires': [{'post.decoder': 'ctc'}, {'post.decoder': 'ctc+edit'}]}
                       for a in (4, 12, 24)},
    'lexicon_params': {'ratio_60': {'set': {'post.lexicon_max_ratio': 0.6},
                                    'requires': [{'post.lexicon': '*'}, {'post.period_lexicon': '*'}]}},
}


def reduced(spec: dict[str, Any], model: str) -> dict[str, Any]:
    out = {**spec, 'base': {**spec.get('base', {}), 'rec.model': model}, 'stages': []}
    for stage in spec['stages']:
        name = stage['name']
        out['stages'].append({**stage, 'candidates': {**stage['candidates'], **EXTRA.get(name, {})},
                              **({'frozen': True} if name in FROZEN else {})})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('spec', type=Path)
    parser.add_argument('model')
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    payload = reduced(json.loads(args.spec.read_text(encoding='utf-8')), args.model)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
