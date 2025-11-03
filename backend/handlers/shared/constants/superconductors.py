"""
Superconductor material constants and reference data.

This module contains structural and electronic properties of known
superconducting materials, particularly cuprate high-Tc superconductors.

CRITICAL: C-axis/apical-oxygen correlation is FAMILY-SPECIFIC:
    - 214 (La₂CuO₄ T-phase): ↑c ↔ apical retained; ↓c ↔ apical loss (T→T′)
    - 123 (YBCO): c tracks CHAIN oxygen; apical on planes persist; deoxygenation INCREASES c
    - Bi-2212/2223: Large c from BiO; apical present; c changes reflect stacking/intercalation
    - T′ (NCCO): No apical O by design; c reorganization via interstitial oxygen
    - Infinite-layer: No apical O; octahedral stability logic not applicable

Key data sources:
    - Keimer et al., Nature 518, 179 (2015) - Cuprate physics review
    - Pavarini et al., Phys. Rev. Lett. 87, 047003 (2001) - Apical O & band structure
    - Ohta et al., Phys. Rev. B 43, 2968 (1991) - Apical O distance vs Tc
    - Avella & Guarino, Phys. Rev. B 105, 014512 (2022) - Electron-doped T′ cuprates, annealing
    - Yamamoto et al., Physica C 470, 1383 (2010) - T vs T′ phases
    - YBCO oxygen content: web.njit.edu/~tyson/supercon_papers/Oxygen_Content_vs_c-axis.pdf
"""

# ============================================================================
# Cuprate Superconductor Structural Data
# ============================================================================

CUPRATE_DATA = {
    "La2CuO4": {
        "typical_c": 13.15,  # Å (T phase, apical O present)
        "typical_c_ortho": 13.13,  # Å (low-T orthorhombic)
        "typical_c_tprime": 12.55,  # Å (T′ phase, no apical O)
        "coordination": "elongated octahedral",
        "apical_distance": 2.4,     # Å (typical)
        "planar_distance": 1.90,    # Å (typical)
        "t_c_max": 40.0,            # K (LSCO x≈0.15)
        "note": "T′ phase (no apical O) exhibits c ≈ 12.5 Å after reduction; c shrinks with apical-O removal.",
        "structure_type": "K2NiF4 (214)",
        "space_group": "I4/mmm / Bmab",
        "family": "214",
        "c_axis_driver": "apical_oxygen",  # What controls c-axis changes
        "c_increase_means": "apical_retention",  # Physical meaning of c increase
    },
    "YBa2Cu3O7": {
        "typical_c": 11.68,  # Å (orthorhombic, δ≈0, O7, superconducting)
        "typical_c_tetra": 11.82,  # Å (tetragonal, δ≈1, O6, insulating)
        "coordination": "square pyramidal (planes) + Cu–O chains",
        "apical_distance": 2.30,   # Å (typical, on plane sites)
        "planar_distance": 1.93,   # Å (typical)
        "chain_distance": 1.94,    # Å (typical)
        "t_c_max": 92.0,           # K
        "note": "Chain oxygen ordering controls hole doping and superconductivity. IMPORTANT: deoxygenation INCREASES c (opposite to 214-type).",
        "structure_type": "123",
        "space_group": "Pmmm / P4/mmm",
        "family": "123",
        "c_axis_driver": "chain_oxygen",  # CHAIN oxygen, NOT apical
        "c_increase_means": "chain_depletion",  # c increases when chains lose O
    },
    "Bi2Sr2CaCu2O8": {
        "typical_c": 30.86,  # Å (Bi-2212, range 30.6-30.9 Å)
        "c_range": (30.6, 30.9),  # Å, typical spread with oxygen content/stacking faults
        "coordination": "square pyramidal",
        "apical_distance": 2.4,  # Å (typical)
        "planar_distance": 1.92, # Å (typical)
        "t_c_max": 95.0,         # K
        "note": "Incommensurate BiO modulation; two CuO2 planes per cell. C-axis varies with oxygen content and stacking faults.",
        "structure_type": "Bi-2212",
        "space_group": "I4/mmm (average, modulated)",
        "family": "Bi-22n",
        "c_axis_driver": "BiO_stacking",
        "c_increase_means": "stacking_intercalation",
    },
    "Bi2Sr2Ca2Cu3O10": {
        "typical_c": 37.1,  # Å (Bi-2223, ~37.0 Å)
        "coordination": "square pyramidal",
        "apical_distance": 2.4,
        "planar_distance": 1.92,
        "t_c_max": 110.0,  # K
        "note": "Three CuO2 planes; higher Tc than Bi-2212.",
        "structure_type": "Bi-2223",
        "space_group": "I4/mmm (modulated)",
        "family": "Bi-22n",
        "c_axis_driver": "BiO_stacking",
        "c_increase_means": "stacking_intercalation",
    },
    "Tl2Ba2Ca2Cu3O10": {
        "typical_c": 35.9,  # Å (Tl-2223)
        "coordination": "square pyramidal",
        "t_c_max": 125.0,  # K
        "note": "Three CuO2 planes; TlO reservoir layers.",
        "structure_type": "Tl-2223",
        "space_group": "I4/mmm",
        "family": "Tl-22n",
        "c_axis_driver": "TlO_stacking",
        "c_increase_means": "stacking_intercalation",
    },
    "HgBa2Ca2Cu3O8": {
        "typical_c": 15.78,  # Å (Hg-1223, range 15.76-15.82 Å)
        "c_range": (15.76, 15.82),  # Å, typical spread in high-quality samples
        "coordination": "square pyramidal",
        "t_c_max": 135.0,  # K (ambient, 134-135 K in best samples), 164 K under pressure
        "note": "Ambient-pressure Tc record holder (≈134–135 K); Tc ≈160–164 K under pressure.",
        "structure_type": "Hg-1223",
        "space_group": "P4/mmm",
        "family": "Hg-12n",
        "c_axis_driver": "HgO_stacking",
        "c_increase_means": "stacking_intercalation",
    },
    "HgBa2CuO4+δ": {  # Hg-1201
        "typical_c": 9.5,   # Å
        "t_c_max": 97.0,    # K (optimal)
        "structure_type": "Hg-1201",
        "space_group": "P4/mmm",
        "note": "Single CuO2 layer; clean tetragonal structure, widely used model system.",
        "family": "Hg-12n",
        "c_axis_driver": "HgO_stacking",
        "c_increase_means": "stacking_intercalation",
    },
    "HgBa2CaCu2O6+δ": {  # Hg-1212
        "typical_c": 12.5,  # Å (typical range 12.5–12.7 depending on δ)
        "t_c_max": 127.0,   # K (reported in high quality samples)
        "structure_type": "Hg-1212",
        "space_group": "P4/mmm",
        "note": "Double-layer Hg cuprate; high Tc without the Bi/Tl modulations.",
        "family": "Hg-12n",
        "c_axis_driver": "HgO_stacking",
        "c_increase_means": "stacking_intercalation",
    },
    # Electron-doped T′ cuprates (no apical oxygen by design)
    "Nd2CuO4": {
        "typical_c": 12.07,  # Å (T′ phase, no apical O)
        "coordination": "square planar",
        "planar_distance": 1.96,  # Å (typical)
        "t_c_max": 24.0,  # K (electron-doped with Ce)
        "note": "T′ structure has NO apical oxygen by design. Annealing reorganizes interstitial oxygen.",
        "structure_type": "T′",
        "space_group": "I4/mmm",
        "family": "T_prime",
        "c_axis_driver": "interstitial_oxygen",
        "c_increase_means": "ambiguous",  # Can go either direction with annealing
    },
    # Infinite-layer cuprates (no apical oxygen, minimal c-axis)
    "CaCuO2": {
        "typical_c": 3.2,  # Å (infinite-layer)
        "coordination": "square planar",
        "planar_distance": 1.93,  # Å (typical)
        "t_c_max": None,  # Not superconducting at ambient (requires high pressure or doping)
        "note": "Infinite-layer structure with NO apical oxygen. Octahedral stability logic not applicable.",
        "structure_type": "infinite-layer",
        "space_group": "P4/mmm",
        "family": "infinite_layer",
        "c_axis_driver": "not_applicable",
        "c_increase_means": "not_applicable",
    },
    "Sr0.9La0.1CuO2": {
        "typical_c": 3.4,  # Å (infinite-layer, doped)
        "coordination": "square planar",
        "planar_distance": 1.95,  # Å (typical)
        "t_c_max": 43.0,  # K (thin film, La-doped)
        "note": "Infinite-layer superconductor (thin films). No apical oxygen.",
        "structure_type": "infinite-layer",
        "space_group": "P4/mmm",
        "family": "infinite_layer",
        "c_axis_driver": "not_applicable",
        "c_increase_means": "not_applicable",
    },
}

# ============================================================================
# Family-Specific C-Axis Behavior Rules
# ============================================================================

FAMILY_C_AXIS_RULES = {
    "214": {
        "description": "La₂CuO₄-type (K₂NiF₄ structure)",
        "c_axis_meaning": "Apical oxygen presence",
        "trend": "larger_c_means_apical_retention",
        "mechanism": "T-phase (c≈13.15 Å) has apical O; T′-phase (c≈12.55 Å) lacks apical O",
        "applicable_to_octahedral_stability": True,
        "citations": [
            "Yamamoto et al., Physica C 470, 1383 (2010): T vs T′ phases",
            "RSC TA (2022): Effect of structure on oxygen diffusivity",
        ],
    },
    "123": {
        "description": "YBa₂Cu₃O₇₋δ-type",
        "c_axis_meaning": "Chain oxygen content (NOT apical on planes)",
        "trend": "larger_c_means_chain_depletion",
        "mechanism": "O₇ (c≈11.66 Å, superconducting) → O₆ (c≈11.84 Å, insulating) with deoxygenation",
        "applicable_to_octahedral_stability": False,  # Chain oxygen, not apical
        "citations": [
            "Jorgensen et al., Phys. Rev. B 41, 1863 (1990): YBCO structure vs oxygen content",
            "web.njit.edu/~tyson/supercon_papers/Oxygen_Content_vs_c-axis.pdf",
        ],
    },
    "Bi-22n": {
        "description": "Bi₂Sr₂Ca(n₋₁)CuₙO(2n+4+δ)-type",
        "c_axis_meaning": "BiO layer stacking and intercalation",
        "trend": "large_c_from_BiO_blocks",
        "mechanism": "BiO bilayers impose large c-axis; apical oxygen present; modulation complicates c",
        "applicable_to_octahedral_stability": False,  # Large c dominated by BiO, not apical
        "citations": [
            "RSC Adv. 2, 239 (2012); RSC 2013: Bi-2212/2223 c-axis",
            "AIP Adv. (2018): Bi₂Sr₂CaCu₂O₈₊ₓ thin films",
        ],
    },
    "Tl-22n": {
        "description": "Tl₂Ba₂Ca(n₋₁)CuₙO(2n+4)-type",
        "c_axis_meaning": "TlO layer stacking",
        "trend": "large_c_from_TlO_blocks",
        "mechanism": "Similar to Bi-22n but less disorder; apical oxygen present",
        "applicable_to_octahedral_stability": False,
        "citations": ["arXiv:2301.08313: Tl-2223"],
    },
    "Hg-12n": {
        "description": "HgBa₂Ca(n₋₁)CuₙO(2n+2+δ)-type",
        "c_axis_meaning": "HgO layer stacking",
        "trend": "compact_c_from_HgO",
        "mechanism": "HgO provides minimal blocking; apical oxygen present; cleanest tetragonal",
        "applicable_to_octahedral_stability": False,
        "citations": [
            "RSC Adv. 12, 32700 (2022): Hg-1223",
            "arXiv:2401.17079: Hg-1223 under pressure",
        ],
    },
    "T_prime": {
        "description": "Nd₂CuO₄-type (T′ structure)",
        "c_axis_meaning": "Interstitial oxygen reorganization",
        "trend": "no_apical_by_design",
        "mechanism": "Square planar CuO₄; annealing changes interstitial O; c can move either direction depending on starting oxygen state",
        "applicable_to_octahedral_stability": False,  # No apical O to stabilize/destabilize
        "citations": [
            "Avella & Guarino, Phys. Rev. B 105, 014512 (2022): Electron-doped NCCO annealing",
            "Matsumoto et al., Physica C 469, 940 (2009): T′ reduction dependence of superconductivity",
        ],
    },
    "infinite_layer": {
        "description": "CaCuO₂-type (infinite-layer)",
        "c_axis_meaning": "Minimal stacking, no apical oxygen",
        "trend": "tiny_c_no_apical",
        "mechanism": "CuO₂ planes with no apical O; c ≈ 3.2–3.4 Å",
        "applicable_to_octahedral_stability": False,  # No octahedral coordination
        "citations": ["Standard cuprate crystallography texts"],
    },
}

# ============================================================================
# Cuprate Structural Trends and Rules (Legacy, now family-specific above)
# ============================================================================

C_AXIS_APICAL_OXYGEN_CORRELATION = {
    "rule": "larger_c_retains_apical",
    "description": (
        "In K₂NiF₄-type (214) cuprates, oxygen reduction removes apical oxygen and decreases c; "
        "larger c correlates with retained apical oxygen and octahedral elongation. "
        "THIS RULE IS FAMILY-SPECIFIC. See FAMILY_C_AXIS_RULES for details."
    ),
    "examples": [
        {"phase": "T-La₂CuO₄", "c": 13.15, "apical_present": True,  "coordination": "octahedral"},
        {"phase": "T′-La₂CuO₄ (reduced)", "c": 12.55, "apical_present": False, "coordination": "square planar"},
    ],
    "caveat": "YBCO and other families have DIFFERENT c-axis drivers; use family-specific rules.",
}

# ============================================================================
# Jahn-Teller Distortion
# ============================================================================

JAHN_TELLER_DISTORTION = {
    "cu2_d9": {
        "description": "Cu²⁺ (d⁹) shows Jahn–Teller elongation in octahedral fields",
        "typical_elongation_ratio": 1.25,  # r_apical / r_planar ~ 2.4/1.92
        "energy_gain": "order-of-magnitude tenths of eV",  # Softened per review
        "note": "Planar Cu–O ≈1.90–1.93 Å; apical ≈2.3–2.4 Å (family-dependent). Energy gain depends on compound and method.",
        "citations": [
            "Bersuker, I. B. (2006). The Jahn-Teller Effect. Cambridge University Press.",
            "Pavarini et al., J. Phys. Condens. Matter 16, S4313 (2004): JT in cuprates",
        ],
    }
}

# ============================================================================
# Superconducting Transition Temperatures (Tc)
# ============================================================================

TC_MAX_VALUES = {
    "La2CuO4": 40.0,       # LSCO (x≈0.15)
    "YBa2Cu3O7": 92.0,
    "Bi2Sr2CaCu2O8": 95.0,
    "Bi2Sr2Ca2Cu3O10": 110.0,
    "Tl2Ba2Ca2Cu3O10": 125.0,
    "HgBa2Ca2Cu3O8": 135.0,  # ambient (134–135 K)
    "HgBa2CuO4+δ": 97.0,
    "HgBa2CaCu2O6+δ": 127.0,
    "Nd2CuO4": 24.0,  # electron-doped with Ce
    "CaCuO2": None,  # Not SC at ambient
    "Sr0.9La0.1CuO2": 43.0,  # thin film
}

# ============================================================================
# Doping Ranges for Cuprates
# ============================================================================

CUPRATE_DOPING_RANGES = {
    "underdoped": {"holes_per_cu": (0.05, 0.16), "characteristics": "pseudogap, rising Tc"},
    "optimal":    {"holes_per_cu": (0.16, 0.19), "characteristics": "maximum Tc (p* is family-dependent)"},
    "overdoped":  {"holes_per_cu": (0.19, 0.27), "characteristics": "Fermi-liquid-like, decreasing Tc"},
}

# ============================================================================
# Other Superconductor Classes (for future expansion)
# ============================================================================

FE_BASED_SC = {
    "LaFeAsO": { "typical_c": 8.74, "t_c_max": 26.0, "note": "1111 family, F-doped discovery" },
    "BaFe2As2": { "typical_c": 13.02, "t_c_max": 38.0, "note": "122 family; K-doped" },
    "SmFeAsO": { "typical_c": 8.50, "t_c_max": 55.0, "note": "1111 family record Tc (F-doped)" },
}

MGB2_DATA = {
    "typical_c": 3.524, "typical_a": 3.086, "t_c": 39.0,
    "note": "Two-gap phonon-mediated superconductor (AlB₂ structure).",
    "structure_type": "AlB2-type",
    "citations": ["Nature 410, 63 (2001): MgB₂ discovery"],
}

# ============================================================================
# Curated Literature References
# ============================================================================

LITERATURE = [
    "Keimer et al., Nature 518, 179 (2015) – cuprate review",
    "Pavarini et al., Phys. Rev. Lett. 87, 047003 (2001) – apical O & band structure / t′ correlation",
    "Ohta et al., Phys. Rev. B 43, 2968 (1991) – Tc vs apical O distance across families",
    "Avella & Guarino, Phys. Rev. B 105, 014512 (2022) – electron-doped NCCO; annealing reorganizes oxygen",
    "Yamamoto et al., Physica C 470, 1383 (2010) – T vs T′ phases in La₂CuO₄",
    "Matsumoto et al., Physica C 469, 940 (2009) – T′ reduction dependence / c-axis vs oxygen",
    "Jorgensen et al., Phys. Rev. B 41, 1863 (1990) – YBCO structure vs oxygen content",
    "YBCO oxygen content: web.njit.edu/~tyson/supercon_papers/Oxygen_Content_vs_c-axis.pdf",
    "Bi-2212/Bi-2223 c-axis: RSC Adv. 2, 239 (2012); AIP Adv. (2018)",
    "Tl-2223 c-axis: arXiv:2301.08313",
    "Hg-1223: RSC Adv. 12, 32700 (2022); arXiv:2401.17079 (pressure response)",
    "MgB₂: Nature 410, 63 (2001); standard a,c lattice values",
    "Bersuker, I. B. (2006). The Jahn-Teller Effect. Cambridge University Press.",
]

