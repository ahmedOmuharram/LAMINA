CUPRATE_DATA = {
    "La2CuO4": {
        "typical_c": 13.15,  # Å (T phase, apical O present)
        "typical_c_ortho": 13.13,  # Å (low-T orthorhombic)
        "coordination": "elongated octahedral",
        "apical_distance": 2.4,     # Å (typical)
        "planar_distance": 1.90,    # Å (typical)
        "t_c_max": 40.0,            # K (LSCO x≈0.15)
        "note": "T′ phase (no apical O) exhibits c ≈ 12.5 Å after reduction; c shrinks with apical-O removal.",
        "structure_type": "K2NiF4 (214)",
        "space_group": "I4/mmm / Bmab",
    },
    "YBa2Cu3O7": {
        "typical_c": 11.68,  # Å (orthorhombic, δ≈0)
        "typical_c_tetra": 11.82,  # Å (tetragonal, δ≈1)
        "coordination": "square pyramidal (planes) + Cu–O chains",
        "apical_distance": 2.30,   # Å (typical)
        "planar_distance": 1.93,   # Å (typical)
        "chain_distance": 1.94,    # Å (typical)
        "t_c_max": 92.0,           # K
        "note": "Chain oxygen ordering controls hole doping and superconductivity.",
        "structure_type": "123",
        "space_group": "Pmmm / P4/mmm",
    },
    "Bi2Sr2CaCu2O8": {
        "typical_c": 30.86,  # Å (Bi-2212)
        "coordination": "square pyramidal",
        "apical_distance": 2.4,  # Å (typical)
        "planar_distance": 1.92, # Å (typical)
        "t_c_max": 95.0,         # K
        "note": "Incommensurate BiO modulation; two CuO2 planes per cell.",
        "structure_type": "Bi-2212",
        "space_group": "I4/mmm (average, modulated)",
    },
    "Bi2Sr2Ca2Cu3O10": {
        "typical_c": 37.1,  # Å (Bi-2223)
        "coordination": "square pyramidal",
        "apical_distance": 2.4,
        "planar_distance": 1.92,
        "t_c_max": 110.0,  # K
        "note": "Three CuO2 planes; higher Tc than Bi-2212.",
        "structure_type": "Bi-2223",
        "space_group": "I4/mmm (modulated)",
    },
    "Tl2Ba2Ca2Cu3O10": {
        "typical_c": 35.9,  # Å (Tl-2223)
        "coordination": "square pyramidal",
        "t_c_max": 125.0,  # K
        "note": "Three CuO2 planes; TlO reservoir layers.",
        "structure_type": "Tl-2223",
        "space_group": "I4/mmm",
    },
    "HgBa2Ca2Cu3O8": {
        "typical_c": 15.78,  # Å (Hg-1223)
        "coordination": "square pyramidal",
        "t_c_max": 135.0,  # K (ambient), ">160 K under pressure"
        "note": "Ambient-pressure Tc record holder; strong pressure response.",
        "structure_type": "Hg-1223",
        "space_group": "P4/mmm",
    },
    # New, single-layer and double-layer Hg cuprates (handy references)
    "HgBa2CuO4+δ": {  # Hg-1201
        "typical_c": 9.5,   # Å
        "t_c_max": 97.0,    # K (optimal)
        "structure_type": "Hg-1201",
        "space_group": "P4/mmm",
        "note": "Single CuO2 layer; clean tetragonal structure, widely used model system.",
    },
    "HgBa2CaCu2O6+δ": {  # Hg-1212
        "typical_c": 12.5,  # Å (typical range 12.5–12.7 depending on δ)
        "t_c_max": 127.0,   # K (reported in high quality samples)
        "structure_type": "Hg-1212",
        "space_group": "P4/mmm",
        "note": "Double-layer Hg cuprate; high Tc without the Bi/Tl modulations.",
    },
}

C_AXIS_APICAL_OXYGEN_CORRELATION = {
    "rule": "larger_c_retains_apical",
    "description": (
        "In K2NiF4-type cuprates, oxygen reduction removes apical oxygen and decreases c; "
        "larger c correlates with retained apical oxygen and (often) octahedral elongation."
    ),
    "examples": [
        {"phase": "T-La2CuO4", "c": 13.15, "apical_present": True,  "coordination": "octahedral"},
        {"phase": "T′-La2CuO4 (reduced)", "c": 12.55, "apical_present": False, "coordination": "square planar"},
    ],
    # refs moved to module docstring (see below)
}

JAHN_TELLER_DISTORTION = {
    "cu2_d9": {
        "description": "Cu²⁺ (d⁹) shows Jahn–Teller elongation in octahedral fields",
        "typical_elongation_ratio": 1.25,  # r_apical / r_planar ~ 2.4/1.92
        "energy_gain": "≈0.5–1.0 eV",
        "note": "Planar Cu–O ≈1.90–1.93 Å; apical ≈2.3–2.4 Å (family-dependent).",
    }
}

TC_MAX_VALUES = {
    "La2CuO4": 40.0,       # LSCO (x≈0.15)
    "YBa2Cu3O7": 92.0,
    "Bi2Sr2CaCu2O8": 95.0,
    "Bi2Sr2Ca2Cu3O10": 110.0,
    "Tl2Ba2Ca2Cu3O10": 125.0,
    "HgBa2Ca2Cu3O8": 135.0,  # ambient
    "HgBa2CuO4+δ": 97.0,
    "HgBa2CaCu2O6+δ": 127.0,
}

CUPRATE_DOPING_RANGES = {
    "underdoped": {"holes_per_cu": (0.05, 0.16), "characteristics": "pseudogap, rising Tc"},
    "optimal":    {"holes_per_cu": (0.16, 0.19), "characteristics": "maximum Tc (p*~family-dependent)"},
    "overdoped":  {"holes_per_cu": (0.19, 0.27), "characteristics": "Fermi-liquid-like, decreasing Tc"},
}

# Other classes (expanded)
FE_BASED_SC = {
    "LaFeAsO": { "typical_c": 8.74, "t_c_max": 26.0, "note": "1111 family, F-doped discovery" },
    "BaFe2As2": { "typical_c": 13.02, "t_c_max": 38.0, "note": "122 family; K-doped" },
    "SmFeAsO": { "typical_c": 8.50, "t_c_max": 55.0, "note": "1111 family record Tc (F-doped)" },
}

MGB2_DATA = {
    "typical_c": 3.524, "typical_a": 3.086, "t_c": 39.0,
    "note": "Two-gap phonon-mediated superconductor (AlB2 structure).",
    "structure_type": "AlB2-type",
}

# Suggested module-level docstring refs (curated)
LITERATURE = [
    "Keimer et al., Nature 518, 179 (2015) – cuprate review",
    "Avella et al., Phys. Rev. B 105, 014512 (2022) – T′ cuprates; annealing & apical oxygen",
    "Matsumoto et al., Physica C 469, 940 (2009) – c-axis vs apical oxygen during reduction",
    "Horio et al., Phys. Rev. Lett. 120, 257001 (2018); arXiv:1710.09028 – electron-doped T′",
    "Ohta et al., Phys. Rev. B 43, 2968 (1991) – Tc vs apical O distance across families",
    "YBCO lattice/phase: Crystals 6, 148 (2016)",
    "Bi-2212/Bi-2223 c-axis: RSC Adv. 2, 239 (2012); RSC 2013 abstract",
    "Tl-2223 c-axis: arXiv:2301.08313",
    "Hg-1223: RSC Adv. 12, 32700 (2022); Einsteinlab blog on pressure record",
    "MgB2: Nature 410, 63 (2001); standard a,c lattice values"
]
