"""Build one immutable production network cache for reuse across analyses."""
import argparse
import json
from pathlib import Path

from network_cache import build_network_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--topology", choices=["original", "clustered"], default="original")
    parser.add_argument("--cluster-params", type=str, default=None,
                        help="JSON object overriding clustered-topology parameters")
    args = parser.parse_args()
    clustered_params = json.loads(args.cluster_params) if args.cluster_params else None
    manifest = build_network_cache(
        args.output, n=100_000, household_mean=5.2, community_mean=30.0,
        community_variance=160.0, seed=args.seed, topology=args.topology,
        clustered_params=clustered_params,
    )
    print(manifest)


if __name__ == "__main__":
    main()
