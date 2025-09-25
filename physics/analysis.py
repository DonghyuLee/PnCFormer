# analysis.py
from __future__ import annotations
import numpy as np

# ---------------------------
# Utilities
# ---------------------------

def _ensure_dp(T_init: np.ndarray, dp: int) -> np.ndarray:
    """
    Ensure T_init has shape (B, DP, 2, 2).
    If given (B, 1, 2, 2), broadcast to (B, DP, 2, 2).
    """
    assert T_init.ndim == 4 and T_init.shape[-2:] == (2, 2), \
        f"T_init shape should be (B, DP, 2, 2), got {T_init.shape}"
    B = T_init.shape[0]
    if T_init.shape[1] == dp:
        return T_init
    if T_init.shape[1] == 1 and dp > 1:
        # broadcast then copy to make it writable
        return np.broadcast_to(T_init, (B, dp, 2, 2)).copy()
    raise ValueError(f"T_init DP mismatch: got {T_init.shape[1]}, expected {dp}")


def batch_matrix_chain_multiply(cell_TMs: np.ndarray, T_init: np.ndarray) -> np.ndarray:
    """
    Vectorized 2x2 chain multiply over (cells, B, DP).

    Parameters
    ----------
    cell_TMs : np.ndarray
        Shape (C, B, DP, 2, 2)
    T_init : np.ndarray
        Shape (B, DP, 2, 2). If DP==1 it will be expanded to DP of cell_TMs.

    Returns
    -------
    np.ndarray
        Final product, shape (B, DP, 2, 2)
    """
    assert cell_TMs.ndim == 5 and cell_TMs.shape[-2:] == (2, 2), \
        f"cell_TMs shape should be (C,B,DP,2,2), got {cell_TMs.shape}"
    C, B, DP, _, _ = cell_TMs.shape
    T = _ensure_dp(T_init, DP)

    # Vectorized multiply: for each cell, T := T @ M  (2x2 batched)
    for c in range(C):
        M = cell_TMs[c]  # (B, DP, 2, 2)

        a = T[..., 0, 0]; b = T[..., 0, 1]
        c_ = T[..., 1, 0]; d = T[..., 1, 1]

        M00 = M[..., 0, 0]; M01 = M[..., 0, 1]
        M10 = M[..., 1, 0]; M11 = M[..., 1, 1]

        out00 = a * M00 + b * M10
        out01 = a * M01 + b * M11
        out10 = c_ * M00 + d * M10
        out11 = c_ * M01 + d * M11

        T[..., 0, 0] = out00; T[..., 0, 1] = out01
        T[..., 1, 0] = out10; T[..., 1, 1] = out11

    return T


# ---------------------------
# Main TMM class
# ---------------------------

class TMM:
    def __init__(self,
                 modulus_A: float, density_A: float,
                 modulus_B: float, density_B: float,
                 batch_size: int,
                 freq_start_hz: float | None = None,
                 freq_stop_hz: float | None = None,
                 freq_step_hz: float | None = None,
                 freq_array_hz: np.ndarray | None = None):
        self.modulus_A = float(modulus_A); self.density_A = float(density_A)
        self.modulus_B = float(modulus_B); self.density_B = float(density_B)
        self.batch_size = int(batch_size)

        if freq_array_hz is not None:
            frow = np.asarray(freq_array_hz, dtype=np.float64).reshape(-1)
            self._freq_is_uniform = False
            self.freq_start_hz = None; self.freq_step_hz = None
        else:
            assert (freq_start_hz is not None and
                    freq_stop_hz  is not None and
                    freq_step_hz  is not None), \
                "Provide (freq_start_hz, freq_stop_hz, freq_step_hz) or freq_array_hz."
            fs, fe, d = float(freq_start_hz), float(freq_stop_hz), float(freq_step_hz)
            # DP = number of points in [fs, fe] with step d
            DP = int(round((fe - fs) / d)) + 1
            frow = fs + d * np.arange(DP, dtype=np.float64)
            self._freq_is_uniform = True
            self.freq_start_hz = fs; self.freq_step_hz = d

        self.data_points = int(frow.size)
        self.f = np.repeat(frow[None, :], repeats=self.batch_size, axis=0)

    def get_freq(self, x: np.ndarray) -> np.ndarray:
        if not self._freq_is_uniform:
            raise NotImplementedError("Non-uniform grid: use self.f directly.")
        return self.freq_start_hz + self.freq_step_hz * (self.data_points * x - 1.0)

    def KL(self, modulus: float, density: float, freq: np.ndarray, length: np.ndarray) -> np.ndarray:
        """
        k*L for longitudinal waves: k = omega / c,  c = sqrt(E/rho)
        freq: (B,DP) [Hz], length: broadcastable to (B,DP) [m]
        """
        velocity = np.sqrt(modulus / density)                     # scalar
        wavenumber = 2.0 * np.pi * freq / velocity                # (B,DP)
        return wavenumber * length                                # (B,DP)

    def Impedance(self, modulus: float, density: float) -> float:
        return float(np.sqrt(density * modulus))

    # ---------- matrices per layer ----------

    def TM(self, modulus: float, density: float, length: np.ndarray) -> np.ndarray:
        """
        Transfer Matrix for a single homogeneous layer.

        Returns complex128 array of shape (B, DP, 2, 2)
        """
        KL_val = self.KL(modulus, density, self.f, length)        # (B,DP)
        cos_KL = np.cos(KL_val)
        sin_KL = np.sin(KL_val)
        Im = self.Impedance(modulus, density)                     # scalar

        TM_real = np.empty((self.batch_size, self.data_points, 2, 2), dtype=np.float64)
        TM_real[:, :, 0, 0] = cos_KL
        TM_real[:, :, 1, 1] = cos_KL
        TM_real[:, :, 0, 1] = 0.0
        TM_real[:, :, 1, 0] = 0.0

        TM_imag = np.empty((self.batch_size, self.data_points, 2, 2), dtype=np.float64)
        TM_imag[:, :, 0, 0] = 0.0
        TM_imag[:, :, 1, 1] = 0.0
        TM_imag[:, :, 0, 1] = sin_KL / Im
        TM_imag[:, :, 1, 0] = Im * sin_KL

        return (TM_real + 1j * TM_imag).astype(np.complex128, copy=False)

    def PM(self, modulus: float, density: float) -> np.ndarray:
        """
        Propagation matrix (frequency-dependent). Shape: (B, DP, 2, 2)
        """
        omega = 2.0 * np.pi * self.f                               # (B,DP)
        Z = self.Impedance(modulus, density)                       # scalar

        PM_real = np.zeros((self.batch_size, self.data_points, 2, 2), dtype=np.float64)

        PM_imag = np.empty((self.batch_size, self.data_points, 2, 2), dtype=np.float64)
        PM_imag[:, :, 0, 0] = omega
        PM_imag[:, :, 0, 1] = omega
        PM_imag[:, :, 1, 0] = -omega * Z
        PM_imag[:, :, 1, 1] =  omega * Z

        return (PM_real + 1j * PM_imag).astype(np.complex128, copy=False)

    # ---------- helpers for supercell ----------

    def _build_cell_TMs(self,
                        length_A: np.ndarray,
                        length_B: np.ndarray,
                        length_C: np.ndarray,
                        length_D: np.ndarray,
                        case: dict) -> np.ndarray:
        """
        Build per-cell transfer matrices (C, B, DP, 2, 2),
        filling with BA := TM_B @ TM_A, and replacing defects with D1A/D2A as needed.
        """
        TM_A = self.TM(self.modulus_A, self.density_A, length_A)  # (B,DP,2,2)
        TM_B = self.TM(self.modulus_B, self.density_B, length_B)  # (B,DP,2,2)

        BA = TM_B @ TM_A                                          # (B,DP,2,2)
        defects = list(case.get("defects", []))
        C = int(case.get("cells", 0))

        # Fill all cells with BA, then overwrite defects
        cell_TMs = np.broadcast_to(BA[None, ...], (C, *BA.shape)).copy()

        if len(defects) == 2:
            D1A = self.TM(self.modulus_B, self.density_B, length_C) @ TM_A
            D2A = self.TM(self.modulus_B, self.density_B, length_D) @ TM_A
            i1, i2 = defects[0] - 1, defects[1] - 1               # 0-based
            cell_TMs[i1] = D1A
            cell_TMs[i2] = D2A
        elif len(defects) == 1:
            DA = self.TM(self.modulus_B, self.density_B, length_C) @ TM_A
            i = defects[0] - 1
            cell_TMs[i] = DA
        # else: no defects

        return cell_TMs  # (C,B,DP,2,2)

    # ---------- public APIs ----------

    def get_dispersion_relation_unitcell(self,
                                         length_A: np.ndarray,
                                         length_B: np.ndarray) -> np.ndarray:
        """
        Dispersion of one AB unitcell (k·a) via trace(T) = 2 cos(k a)
        Returns shape (B, DP)
        """
        TM_A = self.TM(self.modulus_A, self.density_A, length_A)
        TM_B = self.TM(self.modulus_B, self.density_B, length_B)
        T = TM_B @ TM_A                                           # (B,DP,2,2)
        trace_T = T[..., 0, 0] + T[..., 1, 1]                     # (B,DP)
        x = np.clip(np.real(trace_T) / 2.0, -1.0, 1.0)
        dr = np.arccos(x)                                         # (B,DP)
        return dr

    def get_dispersion_relation_supercell(self,
                                          length_A: np.ndarray,
                                          length_B: np.ndarray,
                                          length_C: np.ndarray,
                                          length_D: np.ndarray,
                                          case: dict) -> np.ndarray:
        """
        Dispersion of a supercell with defects.
        Returns shape (B, DP)
        """
        cell_TMs = self._build_cell_TMs(length_A, length_B, length_C, length_D, case)  # (C,B,DP,2,2)

        # Identity init
        B, DP = self.batch_size, self.data_points
        T_init = np.empty((B, DP, 2, 2), dtype=cell_TMs.dtype)
        T_init[..., 0, 0] = 1.0; T_init[..., 0, 1] = 0.0
        T_init[..., 1, 0] = 0.0; T_init[..., 1, 1] = 1.0

        T = batch_matrix_chain_multiply(cell_TMs[::-1], T_init)   # (B,DP,2,2)
        trace_T = T[..., 0, 0] + T[..., 1, 1]                     # (B,DP)
        x = np.clip(np.real(trace_T) / 2.0, -1.0, 1.0)
        dr = np.arccos(x)
        return dr

    def get_transmittance(self,
                          length_A: np.ndarray,
                          length_B: np.ndarray,
                          length_C: np.ndarray,
                          length_D: np.ndarray,
                          case: dict) -> np.ndarray:
        """
        Power transmittance computed via similarity transform S = PM^{-1} T PM.
        Returns shape (B, DP), real-valued in [0, +inf).
        """
        cell_TMs = self._build_cell_TMs(length_A, length_B, length_C, length_D, case)

        # Identity init
        B, DP = self.batch_size, self.data_points
        T_init = np.empty((B, DP, 2, 2), dtype=cell_TMs.dtype)
        T_init[..., 0, 0] = 1.0; T_init[..., 0, 1] = 0.0
        T_init[..., 1, 0] = 0.0; T_init[..., 1, 1] = 1.0

        T = batch_matrix_chain_multiply(cell_TMs[::-1], T_init)   # (B,DP,2,2)

        # Similarity transform with medium B
        PM = self.PM(self.modulus_B, self.density_B)              # (B,DP,2,2)
        PM_inv = np.linalg.inv(PM)                                # (B,DP,2,2)
        SM = PM_inv @ T @ PM                                      # (B,DP,2,2)

        det_SM = np.linalg.det(SM)                                # (B,DP) complex
        SM_11  = SM[..., 1, 1]                                    # (B,DP) complex

        # Avoid division by 0
        eps = 1e-30
        tr = (np.abs(det_SM) / np.maximum(np.abs(SM_11), eps)) ** 2
        return tr  # (B,DP) real-valued