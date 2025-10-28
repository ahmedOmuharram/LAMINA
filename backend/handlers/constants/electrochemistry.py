"""
Electrochemistry and battery constants: diffusion barriers and structure defaults.

Literature-based values for ion diffusion in battery materials.
DO NOT change these values without proper literature citations.
"""

# ============================================================================
# Known Ion Diffusion Barriers (eV)
# ============================================================================

# Known priors (literature benchmarks in eV)
# Key format: (host_material, ion_species)
KNOWN_DIFFUSION_BARRIERS = {
    # Graphite (in-plane). Stage-specific values from NEB (Persson 2010) + μSR (Umegaki 2017).
    ("C6", "Li"): {
        "Ea": 0.28,  # stage I (LiC6)
        "range": [0.20, 0.35],
        "note": "Graphite stage I (LiC6), in-plane hopping",
        "citations": [
            "Persson et al., Phys. Rev. B 82, 125416 (2010), Table II (≈293 meV)",
            "Umegaki et al., PCCP 19, 19058 (2017): Ea(C6Li)=270(5) meV"
        ],
        "descriptors": {"stage": "I", "path": "in-plane"}
    },
    ("C12", "Li"): {
        "Ea": 0.20,  # stage II (LiC12)
        "range": [0.15, 0.28],
        "note": "Graphite stage II (LiC12), in-plane hopping",
        "citations": [
            "Persson et al., Phys. Rev. B 82, 125416 (2010), Table II (≈218-283 meV)",
            "Umegaki et al., PCCP 19, 19058 (2017): Ea(C12Li)=170(20) meV"
        ],
        "descriptors": {"stage": "II", "path": "in-plane"}
    },
    ("graphite", "Li"): {
        "Ea": 0.22,  # stage unspecified → midpoint between stage I/II
        "range": [0.15, 0.30],
        "note": "Graphite (stage unspecified), in-plane hopping",
        "citations": [
            "Persson et al., PRB 82, 125416 (2010)",
            "Umegaki et al., PCCP 19, 19058 (2017)"
        ],
        "descriptors": {"stage": "unspecified", "path": "in-plane"}
    },
}

# ============================================================================
# Structure-Based Diffusion Barrier Defaults (eV)
# ============================================================================

# Structure-based estimates (generic)
# Used when specific material data is not available
STRUCTURE_DIFFUSION_DEFAULTS = {
    "layered": {
        "Ea": 0.30,
        "range": [0.15, 0.50],
        "confidence": "medium",
        "note": "Layered hosts: graphite in-plane ~0.17–0.30 eV (stage dependent); layered oxides/sulfides typically ~0.3–0.5 eV",
    },
    "1D-channel": {
        "Ea": 0.30,
        "range": [0.15, 0.50],
        "confidence": "medium",
        "note": "1D channels: moderate barriers along channel direction",
    },
    "olivine": {
        "Ea": 0.25,
        "range": [0.15, 0.35],
        "confidence": "medium",
        "note": "Olivine structures: 1D channels with typical barriers ~0.2-0.3 eV",
    },
    "3D": {
        "Ea": 0.40,
        "range": [0.20, 0.70],
        "confidence": "low",
        "note": "3D frameworks: higher barriers, more tortuous paths",
    },
    "unknown": {
        "Ea": 0.35,
        "range": [0.10, 0.80],
        "confidence": "low",
        "note": "Structure unknown; wide uncertainty",
    },
}

