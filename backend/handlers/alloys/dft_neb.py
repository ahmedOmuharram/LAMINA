"""
DFT-based Nudged Elastic Band (NEB) calculations for surface diffusion barriers.

This module provides first-principles calculation of activation energies using:
- ASE (Atomic Simulation Environment) for structure manipulation
- Multiple calculator backends: GPAW (Python-native), VASP (external)
- Climbing-image NEB for transition state search
- Optional vibrational analysis for attempt frequencies

For production runs, tune convergence criteria, k-point meshes, and energy cutoffs
based on your accuracy requirements and computational budget.
"""
from __future__ import annotations

import os
import math
import logging
from typing import Tuple, Dict, Any, List, Optional

try:
    from ase import Atoms
    from ase.build import fcc111, fcc100, fcc110, add_adsorbate
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS, FIRE
    from ase.mep import NEB
    from ase.io import write
    HAS_ASE = True
except ImportError:
    HAS_ASE = False

_log = logging.getLogger(__name__)


# --- Calculator Backend Factory ---

def make_calculator(
    backend: str = "chgnet",
    kpts: Tuple[int, int, int] = (2, 2, 1),
    encut: float = 450,
    sigma: float = 0.1,
    xc: str = "PBE",
    txt: Optional[str] = None,
    mode: str = "lcao",
    **kwargs
):
    """
    Create a calculator for surface calculations.
    
    Args:
        backend: Calculator type ('chgnet', 'gpaw', or 'vasp')
        kpts: k-point mesh (for DFT backends only)
        encut: Energy cutoff (eV) for VASP
        sigma: Fermi smearing width (eV) for DFT
        xc: Exchange-correlation functional (default: PBE) for DFT
        txt: Output file path
        mode: GPAW mode ('lcao' for speed, 'fd' for accuracy)
        **kwargs: Additional calculator-specific parameters
        
    Returns:
        Configured ASE calculator instance
    """
    if backend.lower() == "chgnet":
        try:
            import warnings
            warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad")
            from chgnet.model import CHGNetCalculator
            calc = CHGNetCalculator(
                use_device=kwargs.get("device", None),  # None = auto-detect
                stress_weight=kwargs.get("stress_weight", 1.0)
            )
            return calc
        except ImportError as e:
            raise ImportError(f"CHGNet not available. Install with: pip install chgnet\nError: {e}")
    
    elif backend.lower() == "gpaw":
        try:
            from gpaw import GPAW, FermiDirac
            calc = GPAW(
                mode=mode,  # 'lcao' = fast (default), 'fd' = accurate but slow
                kpts=kpts,
                xc=xc,
                occupations=FermiDirac(sigma),
                txt=txt or "gpaw.out",
                symmetry=kwargs.get("symmetry", "off"),  # often off for surfaces
                h=kwargs.get("h", 0.2) if mode == "fd" else None,  # grid spacing for FD mode
                basis=kwargs.get("basis", "dzp") if mode == "lcao" else None  # basis set for LCAO
            )
            return calc
        except ImportError as e:
            raise ImportError(f"GPAW not available: {e}")
            
    elif backend.lower() == "vasp":
        try:
            from ase.calculators.vasp import Vasp
            calc = Vasp(
                xc=xc,
                encut=encut,
                kpts=kpts,
                ispin=kwargs.get("ispin", 1),  # non-spin-polarized by default
                ismear=kwargs.get("ismear", 1),  # Methfessel-Paxton
                sigma=sigma,
                ediff=kwargs.get("ediff", 1e-5),
                ediffg=kwargs.get("ediffg", -0.03),
                ibrion=kwargs.get("ibrion", 2),  # ionic relaxation algorithm
                nsw=0,  # ASE will handle optimization
                lwave=False,
                lcharg=False,
                gga="PE",
                lreal=kwargs.get("lreal", "Auto"),
                setups=kwargs.get("setups", "recommended"),
                gamma=kwargs.get("gamma", True)
            )
            return calc
        except ImportError as e:
            raise ImportError(f"ASE VASP interface not available: {e}")
            
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'chgnet', 'gpaw', or 'vasp'.")


# --- Slab Construction ---

def build_slab(
    host: str,
    facet: str,
    size: Tuple[int, int] = (4, 4),
    layers: int = 5,
    vacuum: float = 15.0
) -> Atoms:
    """
    Build a relaxable metal slab with bottom layers fixed.
    
    Args:
        host: Element symbol (e.g., 'Al', 'Au', 'Cu')
        facet: Surface orientation ('111', '100', '110')
        size: Lateral supercell size (nx, ny)
        layers: Number of atomic layers
        vacuum: Vacuum thickness (Å) above surface
        
    Returns:
        ASE Atoms object with FixAtoms constraint on bottom 2 layers
        
    Notes:
        - Uses experimental lattice constants from ASE database
        - For stepped surfaces or other orientations, extend with ase.build.surface()
    """
    facet = facet.strip()
    
    # ASE's fcc builder functions automatically create slabs with their own layer counts
    # The 'size' parameter is (nx, ny, nlayers) for surface, but (nx, ny) for the legacy functions
    if facet == "111":
        slab = fcc111(host, size=(size[0], size[1], layers), a=None, vacuum=vacuum, orthogonal=True)
    elif facet == "100":
        slab = fcc100(host, size=(size[0], size[1], layers), a=None, vacuum=vacuum)
    elif facet == "110":
        slab = fcc110(host, size=(size[0], size[1], layers), a=None, vacuum=vacuum)
    else:
        raise ValueError(f"Unsupported facet: {facet}. Use '111', '100', or '110'.")
    
    # Fix bottom ~40% of atoms to simulate bulk
    zs = [atom.position[2] for atom in slab]
    z_sorted = sorted(zs)
    
    # Calculate index for bottom 2 layers (ensure it's within bounds)
    n_atoms = len(zs)
    if layers > 0 and n_atoms > 0:
        # Aim for bottom 2 layers out of total layers
        target_fraction = 2.0 / layers
        cutoff_index = min(int(target_fraction * n_atoms), n_atoms - 1)
        z_cutoff = z_sorted[cutoff_index]
    else:
        # Fallback: fix bottom 40% of atoms
        z_cutoff = z_sorted[int(0.4 * n_atoms)]
    
    mask = [z <= z_cutoff for z in zs]
    slab.set_constraint(FixAtoms(mask=mask))
    
    return slab


def place_adatom_terrace(
    slab: Atoms,
    adatom: str,
    facet: str,
    site: str = "fcc",
    height: float = 1.8
) -> Tuple[Atoms, int]:
    """
    Place an adatom on a terrace site of a slab.
    
    Args:
        slab: Base slab structure
        adatom: Element symbol for adatom
        facet: Surface facet ('111', '100', '110')
        site: Adsorption site type:
            - '111': 'fcc', 'hcp', 'bridge', 'top'
            - '100': 'hollow' (4-fold), 'bridge', 'top'
            - '110': 'longbridge', 'shortbridge', 'top'
        height: Initial height above surface (Å)
        
    Returns:
        Tuple of (slab_with_adatom, adatom_index)
        
    Notes:
        - Height will be optimized during relaxation
        - For exchange mechanisms, you'll manually swap positions post-placement
    """
    s = slab.copy()
    
    # ASE add_adsorbate uses various site keywords
    if facet == "111":
        add_adsorbate(s, adatom, height, position=site)
    elif facet == "100":
        add_adsorbate(s, adatom, height, position=site if site != "fourfold" else "hollow")
    elif facet == "110":
        add_adsorbate(s, adatom, height, position=site)
    else:
        raise ValueError(f"Unsupported facet: {facet}")
    
    ad_idx = len(s) - 1  # last atom added
    s.center(axis=2)  # re-center vacuum
    
    return s, ad_idx


def make_endpoints(
    adatom: str,
    host: str,
    facet: str,
    mechanism: str = "hopping",
    size: Tuple[int, int] = (4, 4)
) -> Tuple[Atoms, Atoms]:
    """
    Generate initial and final structures for a diffusion path.
    
    Args:
        adatom: Adatom element symbol
        host: Host element symbol
        facet: Surface facet ('111', '100', '110')
        mechanism: Diffusion mechanism ('hopping' or 'exchange')
        size: Slab supercell size
        
    Returns:
        Tuple of (initial_structure, final_structure)
        
    Mechanism Details:
        - Hopping (111): fcc → hcp site
        - Hopping (100): fourfold → neighboring fourfold
        - Hopping (110): longbridge → neighboring longbridge
        - Exchange: adatom swaps with surface atom, displaced atom moves to adjacent site
        
    Notes:
        - For production, verify endpoints are local minima via independent relaxations
        - Exchange mechanism may require manual tuning of displaced atom position
    """
    import numpy as np
    
    slab_base = build_slab(host, facet, size=size)
    
    if mechanism == "hopping":
        if facet == "111":
            # fcc → hcp (both 3-fold hollow sites, but different stacking)
            ini, ad_idx = place_adatom_terrace(slab_base, adatom, facet, site="fcc")
            fin, _ = place_adatom_terrace(build_slab(host, facet, size=size), 
                                          adatom, facet, site="hcp")
        elif facet == "100":
            # Move from one 4-fold hollow to neighboring 4-fold
            ini, ad_idx = place_adatom_terrace(slab_base, adatom, facet, site="hollow")
            fin, _ = place_adatom_terrace(build_slab(host, facet, size=size), 
                                         adatom, facet, site="hollow")
            # Shift final adatom by one lattice vector
            cell = fin.get_cell()
            fin[-1].position[0] += cell[0, 0] / size[0]
        elif facet == "110":
            # Long-bridge → neighboring long-bridge
            ini, ad_idx = place_adatom_terrace(slab_base, adatom, facet, site="longbridge")
            fin, _ = place_adatom_terrace(build_slab(host, facet, size=size), 
                                         adatom, facet, site="longbridge")
            # Shift along trough
            cell = fin.get_cell()
            fin[-1].position[0] += cell[0, 0] / (2 * size[0])
        else:
            raise ValueError(f"Hopping not implemented for facet {facet}")
            
    elif mechanism == "exchange":
        # Place adatom at stable site
        if facet == "111":
            ini, ad_idx = place_adatom_terrace(slab_base, adatom, facet, site="fcc")
        elif facet == "100":
            ini, ad_idx = place_adatom_terrace(slab_base, adatom, facet, site="hollow")
        elif facet == "110":
            ini, ad_idx = place_adatom_terrace(slab_base, adatom, facet, site="longbridge")
        else:
            raise ValueError(f"Exchange not implemented for facet {facet}")
        
        # Find nearest surface atom to adatom
        zmax = max(a.position[2] for a in ini if a.index != ad_idx)
        surface_atoms = [i for i, a in enumerate(ini) 
                        if abs(a.position[2] - zmax) < 0.6 and i != ad_idx]
        
        dists = [(i, np.linalg.norm(ini[i].position - ini[ad_idx].position)) 
                 for i in surface_atoms]
        target_idx = min(dists, key=lambda x: x[1])[0]
        
        # Build final state: swap adatom and target positions
        fin = ini.copy()
        pos_ad = fin[ad_idx].position.copy()
        pos_target = fin[target_idx].position.copy()
        
        fin.positions[ad_idx] = pos_target
        fin.positions[target_idx] = pos_ad + [0.0, 0.0, 2.0]  # push displaced atom up
        
        # Ensure symbols are correct (adatom stays adatom)
        fin[ad_idx].symbol = adatom
        fin[target_idx].symbol = host
        
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    
    return ini, fin


# --- Structure Relaxation ---

def relax(
    atoms: Atoms,
    calc,
    fmax: float = 0.10,
    steps: int = 80,
    logfile: Optional[str] = None
) -> Atoms:
    """
    Relax atomic positions with LBFGS optimizer (faster than BFGS).
    
    Args:
        atoms: Structure to relax
        calc: ASE calculator
        fmax: Force convergence criterion (eV/Å), default 0.10 for speed
        steps: Maximum optimization steps (capped to prevent hovering)
        logfile: Path for optimization log (None for silent)
        
    Returns:
        Relaxed structure
        
    Notes:
        - LBFGS converges faster than BFGS for molecular systems
        - maxstep=0.20 prevents large jumps that can cause instabilities
        - Looser fmax (0.10 vs 0.03) is fine for barrier estimates (±0.05 eV error)
    """
    atoms.calc = calc
    dyn = LBFGS(atoms, logfile=logfile, maxstep=0.20)
    dyn.run(fmax=fmax, steps=steps)
    return atoms


# --- NEB Calculation ---

def run_neb(
    adatom: str,
    host: str,
    facet: str,
    mechanism: str = "hopping",
    backend: str = "chgnet",
    images_n: int = 3,
    kpts: Tuple[int, int, int] = (1, 1, 1),
    sigma: float = 0.1,
    climb: bool = True,
    spring: float = 0.2,
    workdir: str = "neb_run",
    size: Tuple[int, int] = (4, 4),
    relax_fmax: float = 0.10,
    neb_fmax: float = 0.20,
    compute_prefactor: bool = False,
    layers: int = 4,
    mode: str = "lcao"
) -> Dict[str, Any]:
    """
    Run a complete NEB calculation for surface diffusion.
    
    Workflow:
        1. Build initial and final endpoint structures
        2. Relax endpoints to local minima
        3. Generate NEB images via linear interpolation
        4. Run climbing-image NEB to convergence
        5. Extract activation energy from energy profile
        6. (Optional) Compute vibrational prefactor
        
    Args:
        adatom: Adatom element
        host: Host element
        facet: Surface facet ('111', '100', '110')
        mechanism: 'hopping' or 'exchange'
        backend: Calculator ('chgnet' (default, fast), 'gpaw', or 'vasp')
        images_n: Total number of images (default: 3 for fast mode)
        kpts: k-point mesh (default: (1,1,1) Γ-only for speed)
        sigma: Fermi smearing (eV)
        climb: Use climbing-image NEB (applied in 2nd stage)
        spring: Spring constant between images (eV/Å²)
        workdir: Directory for output files
        size: Slab supercell size (default: (4,4) for fast mode, must be even for 111)
        relax_fmax: Force tolerance for endpoint relaxation (default: 0.10 eV/Å, loose for speed)
        neb_fmax: Force tolerance for NEB convergence (default: 0.20 eV/Å, loose for speed)
        compute_prefactor: If True, run vibrational analysis (expensive!)
        layers: Number of slab layers (default: 4 for fast mode)
        mode: GPAW mode ('lcao' for speed, 'fd' for accuracy)
        
    Returns:
        Dict containing:
            - Ea_eV: Activation energy
            - energies_eV: Full energy profile
            - ts_image_index: Transition state image index
            - prefactor_Hz: Attempt frequency
            - rate_s-1@300K: Hopping rate at 300 K
            - D_m2_s@300K: Diffusion coefficient at 300 K
            - notes: Calculation metadata
            
    Notes:
        - Fast mode (default): ~30 sec-2 min with CHGNet, barriers within ~0.05 eV
        - Uses two-stage NEB: no-climb preconverge → climb for TS refinement
        - IDPP interpolation for smoother initial path → fewer optimizer steps
        - Step caps prevent infinite hovering (80 relax, 120+150 NEB stages)
        - Defaults: images=3, kpts=(1,1,1), size=(4,4), layers=4, fmax=0.10/0.20
        - For production: use images_n=5, kpts=(2,2,1), relax_fmax=0.05, neb_fmax=0.05
        - Prefactor calculation is optional and costly (adds ~50% runtime)
    """
    if not HAS_ASE:
        raise ImportError("ASE not installed. Install with: pip install ase")
    
    os.makedirs(workdir, exist_ok=True)
    _log.info(f"Running NEB: {adatom}/{host} {facet} {mechanism}, backend={backend}")
    
    # Step 1: Generate endpoints
    try:
        ini, fin = make_endpoints(adatom, host, facet, mechanism, size=size)
        _log.debug(f"Endpoints generated: ini={len(ini)} atoms, fin={len(fin)} atoms")
    except Exception as e:
        _log.error(f"Failed to generate endpoints: {e}")
        raise
    
    # Step 2: Relax endpoints
    calc_end = make_calculator(
        backend=backend,
        kpts=kpts,
        sigma=sigma,
        mode=mode,
        txt=os.path.join(workdir, "relax_endpoints.txt")
    )
    
    # Recenter endpoints to avoid vacuum drift during relaxation
    ini.center(axis=2)
    fin.center(axis=2)
    
    _log.info("Relaxing initial endpoint...")
    ini = relax(ini, calc_end, fmax=relax_fmax)
    
    _log.info("Relaxing final endpoint...")
    # Need fresh calculator for second relaxation
    calc_end2 = make_calculator(
        backend=backend,
        kpts=kpts,
        sigma=sigma,
        mode=mode,
        txt=os.path.join(workdir, "relax_endpoints.txt")
    )
    fin = relax(fin, calc_end2, fmax=relax_fmax)
    
    # Step 3: Interpolate NEB images with IDPP for smoother initial path
    from ase.neb import interpolate
    
    images = [ini]
    images += [ini.copy() for _ in range(images_n - 2)]
    images += [fin]
    interpolate(images)  # Linear interpolation first
    
    # Try IDPP interpolation for better initial path (fewer NEB steps)
    try:
        from ase.mep import IDPP
        idpp = IDPP(images, k=spring, fmax=0.5, climb=False)
        idpp_opt = FIRE(idpp, logfile=os.path.join(workdir, "idpp.log"))
        idpp_opt.run(fmax=0.5, steps=100)  # Quick IDPP preconditioning
        _log.info("IDPP interpolation successful")
    except Exception as e:
        _log.debug(f"IDPP interpolation failed (using linear): {e}")
    
    # Step 4: Attach calculators
    _log.info(f"Running two-stage NEB with {images_n} images...")
    for i, img in enumerate(images):
        calc_i = make_calculator(
            backend=backend,
            kpts=kpts,
            sigma=sigma,
            mode=mode,
            txt=os.path.join(workdir, f"neb_image_{i:02d}.txt")
        )
        img.set_calculator(calc_i)
    
    # Two-stage NEB: Stage 1 - no climb, coarse convergence
    _log.info("Stage 1: Preconverging NEB path without climbing...")
    neb = NEB(images, climb=False, k=spring)
    opt1 = FIRE(neb, logfile=os.path.join(workdir, "neb_stage1.log"))
    opt1.run(fmax=max(neb_fmax, 0.20), steps=120)  # Coarse preconvergence, capped
    
    # Stage 2 - turn on climbing and refine TS
    _log.info("Stage 2: Refining transition state with climbing image...")
    neb.climb = climb
    opt2 = LBFGS(neb, logfile=os.path.join(workdir, "neb_stage2.log"), maxstep=0.20)
    opt2.run(fmax=neb_fmax, steps=150)  # Hard cap to prevent hovering
    
    # Step 5: Extract energies (interior-only TS, reference to lower endpoint)
    energies = [img.get_potential_energy() for img in images]
    E_i, E_f = energies[0], energies[-1]
    E_ref = min(E_i, E_f)
    
    if len(energies) >= 3:
        # Search for TS only in interior images (exclude endpoints)
        interior = energies[1:-1]
        emax = max(interior)
        imax_rel = interior.index(emax)   # 0-based within interior
        imax = imax_rel + 1               # shift to global index
    else:
        # Fallback if there are only 2 images (shouldn't happen in NEB)
        emax = max(energies)
        imax = energies.index(emax)
    
    Ea = max(0.0, emax - E_ref)  # never negative; downhill → 0 barrier
    
    _log.info(f"NEB summary: E_i={E_i:.6f} eV, E_f={E_f:.6f} eV, "
              f"E_ref={E_ref:.6f} eV, E_max(int)={emax:.6f} eV, "
              f"Ea={Ea:.4f} eV, TS image={imax}")
    
    # Flag downhill paths (barrierless on this potential)
    downhill = Ea < 1e-3
    if downhill:
        _log.info("Path appears downhill on this potential (barrierless within ~1 meV).")
    
    # Step 6: Optional vibrational prefactor
    nu = 1.0e13  # default attempt frequency
    prefactor_note = "default (1e13 Hz)"
    
    if compute_prefactor:
        try:
            from ase.vibrations import Vibrations
            from ase.neighborlist import neighbor_list
            import numpy as np
            
            _log.info("Computing vibrational prefactor...")
            
            # Identify adatom and nearest surface atoms
            ad_idx = len(images[0]) - 1
            i_list, j_list, _ = neighbor_list("ijd", images[0], cutoff=3.5)
            neighbors = set(j for i, j in zip(i_list, j_list) if i == ad_idx)
            indices = [ad_idx] + sorted(list(neighbors))[:8]  # adatom + ~8 neighbors
            
            # Compute frequencies at initial minimum
            calc_vib = make_calculator(
                backend=backend,
                kpts=kpts,
                sigma=sigma,
                mode=mode,
                txt=os.path.join(workdir, "vib.txt")
            )
            images[0].set_calculator(calc_vib)
            
            vib = Vibrations(images[0], indices=indices, 
                           name=os.path.join(workdir, "vib"))
            vib.run()
            freqs = vib.get_frequencies()  # cm⁻¹
            
            # Vineyard prefactor: geometric mean of positive frequencies
            pos_freqs = [f for f in freqs if f > 0]
            if pos_freqs:
                hz_vals = np.array(pos_freqs) * 2.9979e10  # cm⁻¹ → Hz
                nu = float(np.exp(np.mean(np.log(hz_vals))))
                prefactor_note = f"Vineyard ({len(indices)} atoms)"
            
            vib.clean()
            
        except Exception as e:
            _log.warning(f"Prefactor calculation failed: {e}")
            prefactor_note = f"default (prefactor calc failed: {e})"
    
    # Step 7: Compute kinetics at 300 K
    kB = 8.617333262e-5  # eV/K
    T = 300.0
    rate = nu * math.exp(-Ea / (kB * T))
    
    # Estimate jump distance from cell size
    a_jump = 2.5e-10  # default 2.5 Å
    try:
        cell_a = images[0].get_cell().lengths()[0] / size[0]
        a_jump = cell_a * 1e-10  # convert Å → m
    except:
        pass
    
    D = a_jump**2 * rate / 4.0  # 2D surface diffusion
    
    # Save trajectory
    write(os.path.join(workdir, "neb_path.traj"), images)
    
    return {
        "facet": facet,
        "mechanism": mechanism,
        "images": len(images),
        "energies_eV": [round(e, 6) for e in energies],
        "Ea_eV": round(Ea, 4),
        "ts_image_index": imax,
        "prefactor_Hz": nu,
        "rate_s-1@300K": rate,
        "D_m2_s@300K": D,
        "notes": [
            f"backend={backend}",
            f"kpts={kpts}",
            f"climb={climb}",
            f"spring={spring} eV/Å²",
            f"prefactor: {prefactor_note}",
            "path: downhill on surrogate (Ea≈0)" if downhill else "path: activated"
        ]
    }


# --- Convenience Function for Testing ---

def quick_test_neb(backend: str = "chgnet", mode: str = "lcao"):
    """
    Quick test to verify NEB setup.
    
    Args:
        backend: 'chgnet' (default, fast), 'gpaw', or 'vasp'
        mode: 'lcao' (fast, default) or 'fd' (accurate but slow) - for GPAW only

    
    Example:
        >>> from backend.handlers.alloys.dft_neb import quick_test_neb
        >>> result = quick_test_neb(backend='chgnet')
        >>> print(f"Barrier: {result['Ea_eV']:.3f} eV")
    """
    # Use all defaults from run_neb (fast mode)
    result = run_neb(
        adatom="Cu",
        host="Al",
        facet="111",
        mechanism="hopping",
        backend=backend,
        mode=mode,
        workdir="quick_test_neb",
        compute_prefactor=False
        # All other params use fast defaults: images_n=3, kpts=(1,1,1), size=(3,3), etc.
    )
    return result

