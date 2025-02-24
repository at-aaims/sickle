from .random import RandomSubsampler
from .maxent import MaxentSubsampler
from .phase_space import PhaseSpaceSubsampler


def get_subsampler(data, args, **kwargs):
    """Factory function to select subsampler based on args.method"""
    if args.method == "maxent":
        return MaxentSubsampler(data, args, **kwargs)
    elif args.method == "random":
        return RandomSubsampler(data, args)
    elif args.method == "uips":
        return PhaseSpaceSubsampler(data, args)
    elif args.method == "full":
        raise ValueError("full not yet supported")
    else:
        raise ValueError(f"Unsupported sampling method: {args.method}")
