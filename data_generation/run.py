from __future__ import annotations
import numpy as np
from tqdm import tqdm

from . import (
    Material, SimulationConfig, generate_cases,
    build_tmm, run_case
)

def main():
    # --- materials (default: Copper vs Aluminum) ---
    mat_A = Material(name="Copper",   modulus_GPa=110.0, density=8960.0)
    mat_B = Material(name="Aluminum", modulus_GPa=70.0,  density=2700.0)
    # mat_A = Material(name="Steel",    modulus_GPa=200.0, density=7850.0)
    # mat_B = Material(name="Titanium", modulus_GPa=116.0, density=4500.0)

    config = SimulationConfig(
        mat_A=mat_A,
        mat_B=mat_B,
        n_samples=20000,
        var_low=0.005, var_high=0.100,
        random_seed=7,
        f_start_hz=100.0, f_stop_hz=100000.0, f_step_hz=100.0,
        tmm_batch_size=1, 
        n_jobs=8,
        data_dir=r"data\CA",
        min_cells=4, max_cells=7,
        include_double_defects=True
    )

    # RNG & grids
    np.random.seed(config.random_seed)
    variables = np.random.uniform(low=config.var_low, high=config.var_high, size=(config.n_samples, 4))
    freq_grid_khz = np.arange(config.f_start_hz, config.f_stop_hz + config.f_step_hz, config.f_step_hz) / 1000.0

    # Build solver
    fa = build_tmm(config)

    # Generate cases
    cases = generate_cases(config.min_cells, config.max_cells, include_double=config.include_double_defects)

    # Run all cases
    for case in tqdm(cases, desc="Processing Cases"):
        run_case(case, config, fa, variables, freq_grid_khz, prefix='CA')

if __name__ == "__main__":
    main()