"""
Surface science constants: adsorption and diffusion scaling factors.

These are heuristic anchors based on literature/"rules of thumb":
  E_ads ≈ 0.20–0.30 * E_coh(host)   (broad metal-on-metal scaling)
  E_diff ≈ f_mech * E_ads, where f_mech ≈ 0.12 (hopping) or 0.22–0.30 (exchange)

References:
- Nørskov et al., various surface science studies
- DFT scaling relations for metal surfaces

DO NOT change these values without literature support.
"""

# ============================================================================
# Adsorption Energy Scaling (E_ads / E_coh)
# ============================================================================

# Facet-specific adsorption energy as fraction of cohesive energy
ADS_OVER_COH_111 = 0.22  # close-packed surfaces bind a bit more weakly on average
ADS_OVER_COH_100 = 0.26  # (100) often stronger site competition + exchange tendency
ADS_OVER_COH_110 = 0.28  # open surfaces ~stronger corrugation

# ============================================================================
# Diffusion Barrier Scaling (E_diff / E_ads)
# ============================================================================

# Facet-dependent diffusion/adsorption fractions (representative midpoints)
DIFF_OVER_ADS_111 = 0.12  # hopping-dominated
DIFF_OVER_ADS_100 = 0.24  # exchange-dominated for many metals on fcc(100)
DIFF_OVER_ADS_110 = 0.28  # often even "rougher"/higher

