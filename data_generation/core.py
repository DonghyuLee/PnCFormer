from __future__ import annotations
from typing import List, Tuple, Sequence
import numpy as np
from joblib import Parallel, delayed

from .data_types import Material, Case, SimulationConfig
from .io_utils import filename_for_case, write_h5

def create_sample(mat_A: Material, mat_B: Material, lengths: Sequence[float], case: Case) -> np.ndarray:
    """
    lengths = [length_A, length_B, length_C, length_D]
    - Non-defect cell: A + B(length_B)
    - Defect cell:
        if exactly two total defects in the case: first -> length_C, second -> length_D
        else: length_C
    Returns array [num_layers, 3] of (E[Pa], rho[kg/m^3], length[m]).
    """
    length_A, length_B, length_C, length_D = lengths
    total_cells = case.cells
    defect_cells = set(case.defects)

    layers: List[Tuple[float, float, float]] = []
    defect_count = 0
    for cell in range(1, total_cells + 1):
        # A layer
        layers.append((mat_A.modulus_GPa, mat_A.density, length_A))
        # B layer
        if cell in defect_cells:
            defect_count += 1
            if len(defect_cells) == 2:
                layers.append((mat_B.modulus_GPa, mat_B.density, length_C if defect_count == 1 else length_D))
            else:
                layers.append((mat_B.modulus_GPa, mat_B.density, length_C))
        else:
            layers.append((mat_B.modulus_GPa, mat_B.density, length_B))

    return np.array(layers, dtype=float)

def process_single_sample(var: np.ndarray,
                          case: Case,
                          fa: "TMM",
                          mat_A: Material,
                          mat_B: Material):
    """
    One design point:
      - build input_sample (saved for completeness)
      - compute DR(unitcell), DR(supercell), Transmittance via fa
    """
    input_sample = create_sample(mat_A, mat_B, var, case)

    dr_unitcell  = fa.get_dispersion_relation_unitcell(var[0:1], var[1:2])[0, :]
    dr_supercell = fa.get_dispersion_relation_supercell(var[0:1], var[1:2], var[2:3], var[3:4],
                                                        case={"cells": case.cells, "defects": list(case.defects)})[0, :]
    trans        = fa.get_transmittance(var[0:1], var[1:2], var[2:3], var[3:4],
                                        case={"cells": case.cells, "defects": list(case.defects)})[0, :]
    return input_sample, dr_unitcell, dr_supercell, trans

def run_case(case: Case,
             config: SimulationConfig,
             fa: "TMM",
             variables: np.ndarray,
             freq_grid_khz: np.ndarray,
             prefix: str = "CA") -> None:

    results = Parallel(n_jobs=config.n_jobs, 
                       verbose=0)(
        delayed(process_single_sample)(var, case, fa, config.mat_A, config.mat_B)
        for var in variables
    )

    design_vars, dr_uc, dr_sc, trans = zip(*results)

    # shape to arrays
    design_vars_arr = np.array(design_vars)       # [N, variable-dependent (layer stacks)]
    dr_uc_arr       = np.stack(dr_uc, axis=0)     # [N, F]
    dr_sc_arr       = np.stack(dr_sc, axis=0)     # [N, F]
    trans_arr       = np.stack(trans, axis=0)     # [N, F]
    freqs_arr       = np.repeat(freq_grid_khz[None, :], 
                                repeats=dr_uc_arr.shape[0], axis=0)  # [N, F] (identical rows)

    filename = filename_for_case(prefix=prefix, case=case)
    out_path = f"{config.data_dir}/{filename}"

    meta = {
        "cells": case.cells,
        "defects": list(case.defects),
        "mat_A": {"name": config.mat_A.name, "E_GPa": config.mat_A.modulus_GPa, "rho": config.mat_A.density},
        "mat_B": {"name": config.mat_B.name, "E_GPa": config.mat_B.modulus_GPa, "rho": config.mat_B.density},
        "n_samples": config.n_samples,
        "var_range_m": [config.var_low, config.var_high],
        "freq_hz": {"start": config.f_start_hz, "stop": config.f_stop_hz, "step": config.f_step_hz},
        "tmm": {"batch_size": config.tmm_batch_size},
        "random_seed": config.random_seed
    }

    write_h5(out_path, design_vars_arr, dr_uc_arr, dr_sc_arr, trans_arr, freqs_arr, meta)