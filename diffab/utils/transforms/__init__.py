# Transforms
from .mask import (
    MaskSingleCDR,
    MaskMultipleCDRs,
    MaskAntibody,
    RemoveAntigen,
    RandomRemoveAntigen,
)
from .merge import MergeChains
from .patch import PatchAroundAnchor
from .resize_cdr import (
    ResizeCDR,
    cdr_length_tag,
    normalize_cdr_name,
    normalize_cdr_initial_residue_config,
    parse_cdr_lengths_text,
    plan_cdr_length_variants,
)

# Factory
from ._base import get_transform, Compose
