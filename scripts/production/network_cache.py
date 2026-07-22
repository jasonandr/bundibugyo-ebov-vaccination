"""Persistent CSR network caches for the production C++ engine."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ebola_stochastic_ring import generate_network


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_network_cache(cache_path, *, n, household_mean, community_mean, community_variance, seed):
    """Generate one topology and save CSR adjacency arrays; never overwrite."""
    cache_path = Path(cache_path)
    manifest_path = cache_path.with_suffix(".manifest.json")
    if cache_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite network cache: {cache_path}")
    np.random.seed(seed)
    graph = generate_network(
        n, household_mean=household_mean, community_mean=community_mean,
        community_variance=community_variance,
    )
    offsets = np.zeros(n + 1, dtype=np.int32)
    edge_parts = []
    edge_count = 0
    for node in range(n):
        offsets[node] = edge_count
        neighbours = np.fromiter(graph.neighbors(node), dtype=np.int32)
        edge_parts.append(neighbours)
        edge_count += len(neighbours)
    offsets[n] = edge_count
    edges = np.concatenate(edge_parts) if edge_parts else np.empty(0, dtype=np.int32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, offsets=offsets, edges=edges)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "network_seed": seed, "N": n, "household_mean": household_mean,
        "community_mean": community_mean, "community_variance": community_variance,
        "directed_adjacency_entries": int(len(edges)),
        "network_builder_sha256": sha256(Path(__file__).with_name("ebola_stochastic_ring.py")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class CachedNetwork:
    """Minimal network interface required by the production Python wrapper."""
    def __init__(self, offsets, edges):
        import ebola_stochastic_ring_cpp
        self._n = len(offsets) - 1
        self.cpp_engine = ebola_stochastic_ring_cpp.EbolaEngine(self._n, offsets, edges)

    def number_of_nodes(self):
        return self._n


def load_cached_network(cache_path):
    with np.load(cache_path) as cache:
        return CachedNetwork(cache["offsets"].astype(np.int32), cache["edges"].astype(np.int32))
