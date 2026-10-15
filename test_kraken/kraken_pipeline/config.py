from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PreprocessingConfig:
    contrast_factor: float = 1.8
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    denoise_strength: float = 7.0
    background_sigma: float = 25.0
    upscale_factor: float = 2.0
    deskew_limit: float = 5.0
    deskew_step: float = 0.25
    dewarp_limit: float = 25.0
    nlbin_threshold: float = 0.5


@dataclass(frozen=True)
class ScoringConfig:
    lowercase: bool = False
    strip_punctuation: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    image_dir: Path = Path('Imagens')
    reference_dir: Path = Path('ground_truth')
    output_dir: Path = Path('outputs')
    recognition_model: Path | None = None
    segmentation_model: Path | None = None
    device: str = 'cpu'
    seed: int = 42
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


@dataclass(frozen=True)
class Variant:
    name: str
    operations: tuple[str, ...] = ()
    segmenter: str = 'neural'

    def __post_init__(self) -> None:
        if self.segmenter != 'neural':
            raise ValueError('Este pipeline usa somente segmentação neural. Refaça perfis antigos com neural.')


RAW = Variant('raw')
NEURAL_BASELINE = Variant('kraken_neural', ('kraken_nlbin',))
KRAKEN_BASELINE = NEURAL_BASELINE  # Alias de compatibilidade; não é um controle adicional.
INDIVIDUAL = tuple(Variant(name, (name, 'kraken_nlbin'), 'neural') for name in (
    'grayscale', 'contrast', 'clahe', 'denoise', 'illumination',
    'deskew', 'dewarp', 'upscale', 'otsu',
))
COMBINED = (
    Variant('denoise_clahe', ('denoise', 'clahe', 'kraken_nlbin'), 'neural'),
    Variant('illumination_clahe', ('illumination', 'clahe', 'kraken_nlbin'), 'neural'),
    Variant('deskew_dewarp', ('deskew', 'dewarp', 'kraken_nlbin'), 'neural'),
    Variant('denoise_clahe_upscale', ('denoise', 'clahe', 'upscale', 'kraken_nlbin'), 'neural'),
    Variant('deskew_dewarp_denoise_clahe', ('deskew', 'dewarp', 'denoise', 'clahe', 'kraken_nlbin'), 'neural'),
)
