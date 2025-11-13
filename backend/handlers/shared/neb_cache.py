"""
Persistent disk-based cache for NEB calculations.

Stores NEB results (barriers, paths, metadata) to avoid redundant expensive calculations
across different function calls and sessions. Cache keys are based on physical parameters
that uniquely identify a calculation.
"""
from __future__ import annotations

import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

_log = logging.getLogger(__name__)


class NEBCache:
    """
    Persistent cache for NEB calculation results.
    
    Stores results as JSON files indexed by a hash of the calculation parameters.
    Cache entries include barrier energies, hop distances, structural metadata, and provenance.
    """
    
    def __init__(self, cache_dir: str = "neb_cache"):
        """
        Initialize NEB cache.
        
        Args:
            cache_dir: Directory to store cache files (relative to workspace or absolute)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        _log.info(f"NEB cache initialized at {self.cache_dir.absolute()}")
    
    def _make_key(self, params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key from calculation parameters.
        
        Args:
            params: Dictionary of parameters that define the calculation
                   (material, path_type, stacking, strain, spacing, images, etc.)
        
        Returns:
            Hex string hash of the sorted parameter dictionary
        """
        # Sort keys for deterministic hashing
        sorted_params = json.dumps(params, sort_keys=True)
        hash_obj = hashlib.sha256(sorted_params.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def get(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached NEB result if it exists.
        
        Args:
            params: Calculation parameters
        
        Returns:
            Cached result dictionary or None if not found
        """
        key = self._make_key(params)
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                result = json.load(f)
            _log.info(f"Cache HIT for {params.get('path_key', 'unknown')} (key: {key[:12]}...)")
            return result
        except Exception as e:
            _log.warning(f"Failed to load cache file {cache_file}: {e}")
            return None
    
    def put(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Store NEB result in cache.
        
        Args:
            params: Calculation parameters (used to generate cache key)
            result: NEB result dictionary (Ea_eV, hop_distance_A, notes, etc.)
        """
        key = self._make_key(params)
        cache_file = self.cache_dir / f"{key}.json"
        
        # Store both params and result for debugging
        cache_entry = {
            "params": params,
            "result": result
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_entry, f, indent=2)
            _log.info(f"Cache STORE for {params.get('path_key', 'unknown')} → Ea = {result.get('Ea_eV', 0):.4f} eV (key: {key[:12]}...)")
        except Exception as e:
            _log.warning(f"Failed to write cache file {cache_file}: {e}")
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                _log.warning(f"Failed to delete {cache_file}: {e}")
        
        _log.info(f"Cleared {count} cache entries from {self.cache_dir}")
        return count
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache size, entry count, etc.
        """
        entries = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in entries)
        
        return {
            "num_entries": len(entries),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir.absolute())
        }


# Global cache instance (shared across all handlers)
_global_cache: Optional[NEBCache] = None


def get_neb_cache(cache_dir: str = "neb_cache") -> NEBCache:
    """
    Get or create the global NEB cache instance.
    
    Args:
        cache_dir: Cache directory path
    
    Returns:
        NEBCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = NEBCache(cache_dir=cache_dir)
    return _global_cache


def cache_neb_result(
    material: str,
    path_key: str,
    backend: str,
    images: int,
    kpts: Tuple[int, int, int],
    stacking: Optional[str] = None,
    strain_percent: float = 0.0,
    delta_interlayer_A: float = 0.0,
    theta: float = 0.0,
    defect: str = "none",
    result: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Cache-aware NEB result getter/setter.
    
    If result is None, attempts to retrieve from cache.
    If result is provided, stores it in cache.
    
    Args:
        material: Material identifier (e.g., 'graphite', 'Al', 'Au')
        path_key: Path identifier (e.g., 'stageI_in_gallery', 'AB_BLG_in_gallery_TH-TH')
        backend: Calculator backend ('chgnet', 'gpaw', etc.)
        images: Number of NEB images
        kpts: k-point grid as tuple
        stacking: Stacking type ('AB', 'AA', etc.)
        strain_percent: In-plane strain percentage
        delta_interlayer_A: Interlayer spacing change in Angstroms
        theta: Coverage fraction
        defect: Defect type
        result: If provided, stores this result; if None, attempts retrieval
    
    Returns:
        Cached result if found (when result=None), or None if not cached
    """
    cache = get_neb_cache()
    
    params = {
        "material": material,
        "path_key": path_key,
        "backend": backend,
        "images": images,
        "kpts": kpts,
        "stacking": stacking,
        "strain_percent": float(strain_percent),
        "delta_interlayer_A": float(delta_interlayer_A),
        "theta": float(theta),
        "defect": defect
    }
    
    if result is None:
        # Retrieve from cache
        cached = cache.get(params)
        return cached["result"] if cached else None
    else:
        # Store in cache
        cache.put(params, result)
        return result
