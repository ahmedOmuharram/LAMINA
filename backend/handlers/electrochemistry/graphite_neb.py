"""
NEB calculations for Li ion hopping in graphite/graphene galleries.

Reuses the same ASE NEB infrastructure from alloys.dft_neb but with
graphite-specific endpoint builders for interlayer Li diffusion.
"""
from __future__ import annotations

import os
import math
import logging
from typing import Tuple, Dict, Any, Optional, List

try:
    from ase import Atoms
    from ase.build import graphene
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS, FIRE
    from ase.mep import interpolate, idpp_interpolate, NEB
    from ase.io import write
    HAS_ASE = True
except ImportError:
    HAS_ASE = False

from backend.handlers.shared.neb_cache import cache_neb_result

_log = logging.getLogger(__name__)


# --- Calculator Factory (reused from surface NEB) ---

def make_calculator(backend: str = "chgnet", **kwargs):
    """
    Create a calculator for graphite NEB calculations.
    
    Args:
        backend: 'chgnet' (default, fast) or 'gpaw'
        **kwargs: Additional calculator options
        
    Returns:
        ASE calculator instance
    """
    if backend.lower() == "chgnet":
        try:
            import warnings
            warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad")
            from chgnet.model import CHGNetCalculator
            return CHGNetCalculator(use_device=kwargs.get("device", None))
        except ImportError as e:
            raise ImportError("CHGNet not available. Install with: pip install chgnet") from e
    
    elif backend.lower() == "gpaw":
        try:
            from gpaw import GPAW, FermiDirac
            return GPAW(
                mode=kwargs.get("mode", "lcao"),
                kpts=kwargs.get("kpts", (2, 2, 1)),
                xc=kwargs.get("xc", "PBE"),
                occupations=FermiDirac(kwargs.get("sigma", 0.1)),
                txt=kwargs.get("txt", "gpaw.out"),
                symmetry="off",
                basis=kwargs.get("basis", "dzp") if kwargs.get("mode", "lcao") == "lcao" else None,
                h=kwargs.get("h", 0.2) if kwargs.get("mode", "lcao") != "lcao" else None
            )
        except ImportError as e:
            raise ImportError(f"GPAW not available: {e}") from e
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'chgnet' or 'gpaw'.")


# --- Graphite Structure Builders ---

def _sqrt3_supercell(at: Atoms) -> Atoms:
    """
    Create √3×√3 R30° supercell for Li sublattice in graphite.
    
    The transformation matrix [[1,1,0], [-1,2,0], [0,0,1]] has area factor 3
    and creates the proper Li ordering for LiC₆ stoichiometry.
    """
    import numpy as np
    from ase.build import make_supercell
    
    M = np.array([[1, 1, 0],
                  [-1, 2, 0],
                  [0, 0, 1]], dtype=int)
    return make_supercell(at, M)


def build_graphite_stack(n_layers: int = 4, a: float = 2.46, c: float = 3.35) -> Atoms:
    """
    Build ABAB-stacked graphite with √3×√3 in-plane supercell.
    
    Args:
        n_layers: Number of graphene layers (≥4 for realistic stack)
        a: In-plane lattice constant (Å)
        c: Interlayer spacing (Å)
        
    Returns:
        Graphite stack with 3D PBC (no vacuum)
    """
    import numpy as np
    
    # Start from √3×√3 graphene so 1 Li per layer gives LiC₆
    mono = _sqrt3_supercell(graphene(a=a, vacuum=0.0))
    
    layers = []
    cell = mono.get_cell()
    ab_shift = (1/3) * cell[0] + (2/3) * cell[1]  # AB stacking shift
    
    for k in range(n_layers):
        gk = mono.copy()
        if k % 2 == 1:  # Odd layers get AB shift
            gk.translate(ab_shift)
        gk.translate([0, 0, k * c])
        layers.append(gk)
    
    # Combine all layers
    bulk = layers[0]
    for layer in layers[1:]:
        bulk += layer
    
    # 3D periodic; no vacuum along z
    cell_bulk = bulk.get_cell()
    cell_bulk[2, 2] = n_layers * c
    bulk.set_cell(cell_bulk, scale_atoms=False)
    bulk.set_pbc([True, True, True])
    
    return bulk


def add_li_plane(bulk: Atoms, z_center: float) -> Atoms:
    """
    Add one Li plane at specified z-coordinate in gallery.
    
    Places 1 Li per √3×√3 cell at hollow site for LiC₆ stoichiometry.
    """
    import numpy as np
    
    cell = bulk.get_cell()
    a1, a2 = cell[0], cell[1]
    xy = (1/3) * a1 + (1/3) * a2  # Hollow site position
    
    li_pos = np.array([xy[0], xy[1], z_center])
    Li = Atoms('Li', positions=[li_pos])
    Li.set_cell(cell, scale_atoms=False)
    Li.set_pbc([True, True, True])
    
    return bulk + Li


def build_stage_I(a: float = 2.46, c: float = 3.35, n_layers: int = 4) -> Atoms:
    """
    Build Stage-I lithiated graphite (LiC₆).
    
    Li in every gallery between graphene layers.
    """
    bulk = build_graphite_stack(n_layers=n_layers, a=a, c=c)
    
    # Galleries at midpoints between layers: c/2, 3c/2, ...
    zs = [c * (k + 0.5) for k in range(n_layers - 1)]
    
    for zc in zs:
        bulk = add_li_plane(bulk, zc)
    
    return bulk  # Stoichiometry ~ LiC₆


def build_stage_II(a: float = 2.46, c: float = 3.35, n_layers: int = 4) -> Atoms:
    """
    Build Stage-II lithiated graphite (LiC₁₂).
    
    Li in every other gallery (alternating occupied/empty).
    """
    bulk = build_graphite_stack(n_layers=n_layers, a=a, c=c)
    
    # Galleries at midpoints between layers
    zs = [c * (k + 0.5) for k in range(n_layers - 1)]
    
    # Occupy every other gallery
    for i, zc in enumerate(zs):
        if i % 2 == 0:
            bulk = add_li_plane(bulk, zc)
    
    return bulk  # Stoichiometry ~ LiC₁₂


def build_bilayer_graphene(
    stacking: str = "AB",
    size: Tuple[int, int] = (4, 4),
    a: float = 2.46,
    c: float = 3.35,
    delta_c: float = 0.0,
    eps_inplane: float = 0.0
) -> Atoms:
    """
    Build bilayer graphene with adjustable stacking, interlayer spacing, and strain.
    
    Args:
        stacking: "AB" or "AA" stacking
        size: Lateral supercell size (nx, ny)
        a: In-plane lattice constant (Å)
        c: Base interlayer spacing (Å)
        delta_c: Change in interlayer spacing (Å), e.g., +0.2 = expansion
        eps_inplane: In-plane strain (%), e.g., 2.0 = 2% tensile strain
        
    Returns:
        ASE Atoms object with two graphene layers separated by c + delta_c
        
    Notes:
        - AB stacking: second layer shifted by (1/3, 2/3) in fractional coordinates
        - AA stacking: second layer directly above first (no shift)
        - Includes vacuum padding along z to prevent self-interaction
        - Strain is applied isotropically in x-y plane
    """
    # Build first graphene layer
    g1 = graphene(a=a, vacuum=0.0)
    g1 = g1.repeat((size[0], size[1], 1))
    
    # Build second layer with stacking-dependent shift
    g2 = g1.copy()
    cell = g1.get_cell()
    
    if stacking.upper() == "AB":
        shift_xy = (1/3) * cell[0] + (2/3) * cell[1]  # AB stacking shift
    else:  # "AA"
        shift_xy = 0 * cell[0] + 0 * cell[1]  # No shift for AA
    
    g2.translate(shift_xy)
    
    # Separate layers along z with adjusted interlayer spacing
    effective_c = c + delta_c
    g1.translate([0, 0, +0.5 * effective_c])
    g2.translate([0, 0, -0.5 * effective_c])
    
    # Combine layers
    blg = g1 + g2
    
    # Apply in-plane strain if requested
    cell = blg.get_cell()
    if abs(eps_inplane) > 1e-6:
        strain_factor = 1.0 + eps_inplane / 100.0
        cell[0, :] *= strain_factor
        cell[1, :] *= strain_factor
    
    # Add vacuum padding
    pad = 12.0
    cell[2, 2] = effective_c + pad
    blg.set_cell(cell, scale_atoms=True)  # scale_atoms to apply strain
    blg.center(axis=2)
    blg.set_pbc([True, True, True])
    
    return blg


def place_li_in_gallery(
    at: Atoms,
    site: str = "TH"
) -> Tuple[int, Atoms]:
    """
    Place a single Li atom in the interlayer gallery.
    
    Args:
        at: Bilayer graphene structure
        site: Site type - "TH" (tetrahedral-like) or "H" (hollow-like)
        
    Returns:
        Tuple of (Li atom index, modified Atoms object)
        
    Notes:
        - TH sites: Li over carbon in one layer, hexagon hollow in the other
        - H sites: Li over hollow sites in both layers
    """
    import numpy as np
    
    a_pos = at.get_positions()
    zmin, zmax = a_pos[:, 2].min(), a_pos[:, 2].max()
    zmid = 0.5 * (zmin + zmax)  # Gallery midplane
    
    # In-plane reference position
    c0 = a_pos[0, :].copy()
    cell = at.get_cell()
    e1, e2 = cell[0], cell[1]
    
    # Site-specific offset
    if site.upper() == "TH":
        offs = (1/3) * e1 + (1/3) * e2
    else:  # "H"
        offs = 0.5 * e1 + 0.5 * e2
    
    p = c0 + offs
    p[2] = zmid
    
    # Add Li atom
    li = Atoms("Li", positions=[p])
    li.set_cell(at.get_cell(), scale_atoms=False)
    li.set_pbc([True, True, True])
    
    combo = at + li
    return len(combo) - 1, combo


def mobile_mask_for_gallery(
    at: Atoms,
    li_idx: int,
    radius_A: float = 4.0
) -> List[bool]:
    """
    Create mask for mobile atoms during relaxation.
    
    Args:
        at: Structure with Li in gallery
        li_idx: Index of Li atom
        radius_A: Radius around Li for mobile carbon atoms (Å)
        
    Returns:
        Boolean mask for atoms to FIX (immobile atoms = True)
        
    Notes:
        - Li and nearby C atoms within radius are mobile
        - Distant C atoms are fixed to maintain structure
    """
    import numpy as np
    
    pos = at.get_positions()
    p_li = pos[li_idx]
    dist = np.linalg.norm(pos - p_li, axis=1)
    
    # Mobile if within radius OR the atom is Li
    mobile = (dist <= radius_A)
    mobile[li_idx] = True
    
    # FixAtoms expects a mask for atoms to FIX
    fix_mask = ~mobile
    return list(fix_mask)


def build_endpoints_graphite(
    path_key: str,
    stacking: str = "AB",
    size: Tuple[int, int] = (4, 4),
    delta_interlayer_A: float = 0.0,
    strain_percent: float = 0.0
) -> Tuple[Atoms, Atoms, int, int]:
    """
    Build initial and final structures for a Li hop in graphite gallery.
    
    Handles three cases:
    - Stage-I graphite (LiC₆): Multi-layer with Li in every gallery
    - Stage-II graphite (LiC₁₂): Multi-layer with Li in every other gallery
    - Bilayer graphene (BLG): Simple bilayer with dilute Li
    
    Args:
        path_key: Descriptor of the hopping path
        stacking: Stacking type ("AB" or "AA") for BLG
        size: Lateral supercell size (ignored for Stage-I/II, uses √3×√3)
        delta_interlayer_A: Change in interlayer spacing (Å)
        strain_percent: In-plane strain (%)
        
    Returns:
        Tuple of (initial_atoms, final_atoms, li_idx_ini, li_idx_fin)
    """
    import numpy as np
    
    # Detect scenario type
    is_stage_I = "stageI" in path_key
    is_stage_II = "stageII" in path_key
    
    if is_stage_I or is_stage_II:
        # Build proper multi-layer graphite with Li planes
        c_eff = 3.35 + delta_interlayer_A
        a_eff = 2.46 * (1 + strain_percent / 100.0)
        
        if is_stage_I:
            bulk = build_stage_I(a=a_eff, c=c_eff, n_layers=4)
        else:
            bulk = build_stage_II(a=a_eff, c=c_eff, n_layers=4)
        
        # Find Li atoms and pick the central gallery
        li_indices = [i for i, atom in enumerate(bulk) if atom.symbol == 'Li']
        if len(li_indices) == 0:
            raise ValueError(f"No Li atoms found in {path_key}")
        
        # Pick first Li as initial position
        liA_idx = li_indices[0]
        ini = bulk.copy()
        
        # For final, move this Li to a neighboring site in the Li sublattice
        # Hop vector is NN in the √3×√3 Li lattice: (a1+a2)/√3
        fin = bulk.copy()
        liB_idx = liA_idx
        
        cell = ini.get_cell()
        a1, a2 = cell[0], cell[1]
        # Li sublattice NN hop (within same plane)
        hop_vec = (a1 + a2) / np.sqrt(3)
        fin.positions[liB_idx] += hop_vec
        
    else:
        # Bilayer graphene (BLG) with dilute Li - original logic
        blg = build_bilayer_graphene(
            stacking=stacking,
            size=size,
            delta_c=delta_interlayer_A,
            eps_inplane=strain_percent
        )
        
        # Determine site types from path_key
        if "TH" in path_key:
            siteA, siteB = "TH", "TH"
        else:
            siteA, siteB = "H", "H"
        
        # Place Li at initial site
        liA_idx, ini = place_li_in_gallery(blg, siteA)
        
        # Place Li at final site
        liB_idx, fin = place_li_in_gallery(blg, siteB)
        
        # Shift final Li to nearest neighbor
        cell = ini.get_cell()
        a1, a2 = cell[0], cell[1]
        hop_vec = (a1 + a2) / 3.0
        fin.positions[liB_idx] += hop_vec
    
    return ini, fin, liA_idx, liB_idx


# --- Relaxation ---

def relax(
    at: Atoms,
    calc,
    fmax: float = 0.10,
    steps: int = 80,
    logfile: Optional[str] = None
) -> Atoms:
    """
    Relax structure using LBFGS optimizer.
    
    Args:
        at: Structure to relax
        calc: ASE calculator
        fmax: Force convergence criterion (eV/Å)
        steps: Maximum steps (capped to prevent hovering)
        logfile: Optional log file path
        
    Returns:
        Relaxed structure
    """
    at.calc = calc
    dyn = LBFGS(at, logfile=logfile, maxstep=0.20)
    dyn.run(fmax=fmax, steps=steps)
    return at


# --- Main NEB Runner ---

def run_graphite_neb(
    stacking: str = "AB",
    theta: float = 0.0,
    path_key: str = "AB_BLG_in_gallery_TH-TH",
    images_n: int = 5,
    backend: str = "chgnet",
    kpts: Tuple[int, int, int] = (1, 1, 1),
    workdir: str = "neb_graphite",
    relax_fmax: float = 0.10,
    neb_fmax: float = 0.20,
    size: Tuple[int, int] = (4, 4),
    delta_interlayer_A: float = 0.0,
    strain_percent: float = 0.0,
    defect: str = "none"
) -> Dict[str, Any]:
    """
    Run complete NEB calculation for Li hopping in graphite gallery.
    
    Uses persistent disk-based cache to avoid redundant calculations.
    
    Workflow:
        1. Check persistent cache for existing result
        2. Build initial and final endpoint structures
        3. Set up constraints (fix distant C, relax Li + local C)
        4. Relax endpoints to local minima
        5. Interpolate NEB images (linear + IDPP)
        6. Run two-stage NEB (no-climb → climb)
        7. Extract activation energy and store in cache
        
    Args:
        stacking: Stacking type ("AB" or "AA")
        theta: Coverage parameter (not used in structure, for bookkeeping)
        path_key: Descriptor string for the hop type
        images_n: Total number of NEB images (default: 5)
        backend: Calculator ('chgnet' (default, fast) or 'gpaw')
        kpts: k-point mesh (ignored by CHGNet)
        workdir: Directory for output files
        relax_fmax: Force tolerance for endpoint relaxation (eV/Å)
        neb_fmax: Force tolerance for NEB convergence (eV/Å)
        size: Lateral supercell size (nx, ny)
        delta_interlayer_A: Change in interlayer spacing (Å)
        strain_percent: In-plane strain (%)
        
    Returns:
        Dict containing:
            - Ea_eV: Activation energy
            - energies_eV: Full energy profile
            - ts_image_index: Transition state image index
            - images: Number of images
            - notes: Calculation metadata including constraints_radius, stacking, etc.
            
    Notes:
        - Default images_n=5 provides robust TS identification
        - Uses same two-stage NEB as surface diffusion code
        - IDPP interpolation for smoother initial path
        - Step caps prevent infinite hovering
        - Constraints radius: 4.0 Å around Li for local relaxation
    """
    if not HAS_ASE:
        raise ImportError("ASE not installed. Install with: pip install ase")
    
    # Check persistent cache first
    cached = cache_neb_result(
        material="graphite",
        path_key=path_key,
        backend=backend,
        images=images_n,
        kpts=kpts,
        stacking=stacking,
        strain_percent=strain_percent,
        delta_interlayer_A=delta_interlayer_A,
        theta=theta,
        defect=defect,
        result=None  # Retrieve mode
    )
    
    if cached is not None:
        _log.info(f"Using cached NEB result for {path_key} (Ea = {cached['Ea_eV']:.4f} eV)")
        return cached
    
    os.makedirs(workdir, exist_ok=True)
    _log.info(f"Running NEW graphite NEB: {path_key}, stacking={stacking}, backend={backend}")
    
    # Step 1: Build endpoints with geometry modifiers
    try:
        ini, fin, liA_idx, liB_idx = build_endpoints_graphite(
            path_key=path_key,
            stacking=stacking,
            size=size,
            delta_interlayer_A=delta_interlayer_A,
            strain_percent=strain_percent
        )
        _log.debug(f"Endpoints generated: ini={len(ini)} atoms, fin={len(fin)} atoms")
    except Exception as e:
        _log.error(f"Failed to generate endpoints: {e}")
        raise
    
    # Step 2: Set up constraints (fix distant C, relax Li + local C)
    # Use 4.0 Å radius for better local relaxation
    constraints_radius = 4.0
    ini_fix = FixAtoms(mask=mobile_mask_for_gallery(ini, liA_idx, radius_A=constraints_radius))
    fin_fix = FixAtoms(mask=mobile_mask_for_gallery(fin, liB_idx, radius_A=constraints_radius))
    ini.set_constraint(ini_fix)
    fin.set_constraint(fin_fix)
    
    # Step 3: Relax endpoints
    calc_end = make_calculator(
        backend=backend,
        kpts=kpts,
        txt=os.path.join(workdir, "relax_endpoints.txt")
    )
    
    _log.info("Relaxing initial endpoint...")
    ini = relax(ini, calc_end, fmax=relax_fmax)
    
    _log.info("Relaxing final endpoint...")
    calc_end2 = make_calculator(
        backend=backend,
        kpts=kpts,
        txt=os.path.join(workdir, "relax_endpoints.txt")
    )
    fin = relax(fin, calc_end2, fmax=relax_fmax)
    
    # Step 4: Interpolate NEB images with IDPP
    images: List[Atoms] = [ini] + [ini.copy() for _ in range(images_n - 2)] + [fin]
    
    interpolate(images, apply_constraint=True)
    
    # Try IDPP for better initial path
    try:
        idpp_interpolate(images, traj=os.path.join(workdir, "idpp.traj"))
        _log.info("IDPP interpolation successful")
    except Exception as e:
        _log.debug(f"IDPP interpolation failed (using linear): {e}")
    
    # Step 5: Attach calculators
    _log.info(f"Running two-stage NEB with {images_n} images...")
    for i, img in enumerate(images):
        calc_i = make_calculator(
            backend=backend,
            kpts=kpts,
            txt=os.path.join(workdir, f"neb_image_{i:02d}.txt")
        )
        img.calc = calc_i
    
    # Sanity checks: ensure calculators attached and free degrees of freedom exist
    for i, img in enumerate(images):
        assert img.calc is not None, f"Image {i} missing calculator"
        
        # Count free atoms (not constrained)
        n_free = sum(
            1 for j in range(len(img))
            if not any(
                isinstance(c, FixAtoms) and 
                (getattr(c, "mask", None) is not None and c.mask[j])
                for c in (img.constraints or [])
            )
        )
        assert n_free > 0, f"Image {i} has zero free atoms (all constrained)"
        _log.debug(f"Image {i}: {n_free}/{len(img)} atoms free")
    
    # Two-stage NEB
    _log.info("Stage 1: Preconverging NEB path without climbing...")
    neb = NEB(images, climb=False, k=0.2)
    opt1 = FIRE(neb, logfile=os.path.join(workdir, "neb_stage1.log"))
    opt1.run(fmax=max(neb_fmax, 0.20), steps=120)
    
    _log.info("Stage 2: Refining transition state with climbing image...")
    neb.climb = True
    opt2 = LBFGS(neb, logfile=os.path.join(workdir, "neb_stage2.log"), maxstep=0.20)
    opt2.run(fmax=neb_fmax, steps=150)
    
    # Step 6: Extract energies (interior-only TS, reference to lower endpoint)
    energies = [float(img.get_potential_energy()) for img in images]
    E_i, E_f = energies[0], energies[-1]
    E_ref = min(E_i, E_f)
    
    if len(energies) >= 3:
        # Search for TS only in interior images
        interior = energies[1:-1]
        emax = max(interior)
        imax_rel = interior.index(emax)
        imax = imax_rel + 1  # shift to global index
    else:
        emax = max(energies)
        imax = energies.index(emax)
    
    Ea = max(0.0, emax - E_ref)  # Never negative
    
    _log.info(f"NEB summary: E_i={E_i:.6f} eV, E_f={E_f:.6f} eV, "
              f"E_ref={E_ref:.6f} eV, E_max(int)={emax:.6f} eV, "
              f"Ea={Ea:.4f} eV, TS image={imax}")
    
    # TS sanity check: warn if barrier is suspiciously small (flat/drift path)
    if 0.001 < Ea < 0.005:
        _log.warning(f"Very low barrier (Ea={Ea:.4f} eV). May indicate flat landscape or drift.")
    
    # Flag downhill paths
    downhill = Ea < 1e-3
    if downhill:
        _log.info("Path appears downhill on this potential (barrierless within ~1 meV).")
    
    # Calculate hop distance for provenance
    import numpy as np
    hop_distance_A = np.linalg.norm(fin.positions[liB_idx] - ini.positions[liA_idx])
    
    # Save trajectory
    write(os.path.join(workdir, "neb_path.traj"), images)
    
    result = {
        "Ea_eV": round(Ea, 4),
        "energies_eV": [round(e, 6) for e in energies],
        "ts_image_index": imax,
        "images": len(images),
        "notes": [
            f"backend={backend}",
            f"kpts={tuple(kpts)}",
            f"stacking={stacking}",
            f"path={path_key}",
            f"images={images_n}",
            f"neb_fmax={neb_fmax} eV/Å",
            f"relax_fmax={relax_fmax} eV/Å",
            f"constraints_radius={constraints_radius} Å",
            f"hop_distance={hop_distance_A:.3f} Å",
            f"delta_interlayer={delta_interlayer_A:.3f} Å",
            f"strain={strain_percent:.2f}%",
            "gallery hop",
            "path: downhill on surrogate (Ea≈0)" if downhill else "path: activated"
        ]
    }
    
    # Store in persistent cache
    cache_neb_result(
        material="graphite",
        path_key=path_key,
        backend=backend,
        images=images_n,
        kpts=kpts,
        stacking=stacking,
        strain_percent=strain_percent,
        delta_interlayer_A=delta_interlayer_A,
        theta=theta,
        defect=defect,
        result=result  # Store mode
    )
    
    return result

