"""Build one immutable production network cache for reuse across analyses."""
import argparse
from pathlib import Path

from network_cache import build_network_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    manifest = build_network_cache(
        args.output, n=100_000, household_mean=5.2, community_mean=30.0,
        community_variance=160.0, seed=args.seed,
    )
    print(manifest)


if __name__ == "__main__":
    main()
