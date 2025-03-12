import numpy as np
from .random import RandomSubsampler
from .maxent import MaxentSubsampler
from .phase_space import PhaseSpaceSubsampler
from .stratified import StratifiedSubsampler
from .lhs import LatinHypercubeSubsampler

def get_subsampler(data, args, coords, **kwargs):
    """Factory function to select subsampler based on args.method"""
    if args.method == "maxent":
        return MaxentSubsampler(data, args, coords, **kwargs)
    elif args.method == "random":
        return RandomSubsampler(data, args, coords)
    elif args.method == "uips":
        return PhaseSpaceSubsampler(data, args, coords)
    elif args.method == "stratified":
        return StratifiedSubsampler(data, args, coords)
    elif args.method == "lhs":
        return LatinHypercubeSubsampler(data, args, coords)
    elif args.method == "full":
        return np.arange(data.shape[1])
    else:
        raise ValueError(f"Unsupported sampling method: {args.method}")
