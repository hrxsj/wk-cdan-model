"""Runtime initialization settings shared by every experiment module."""
import os
import secrets


def parse_seeds(value):
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("Initialization values must be a nonempty list of distinct integers")
    if any(value < 0 or value >= 2**31 - 10000 for value in values):
        raise ValueError("Initialization values must be nonnegative and below the RNG limit")
    return values


def _initializations():
    configured = os.environ.get("WKCDAN_RUN_SEEDS", "").strip()
    if configured:
        return parse_seeds(configured)
    values = set()
    while len(values) < 5:
        values.add(secrets.randbelow(2**30))
    return tuple(sorted(values))


DEFAULT_RUN_SEEDS = _initializations()


def resolve_seed(value=None):
    return DEFAULT_RUN_SEEDS[0] if value is None else int(value)
