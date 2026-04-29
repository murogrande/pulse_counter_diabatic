import torch
import pulser
from pulser.backend import EmulationConfig

from pulse_counter_diabatic.rydberg_to_ising import from_rydberg_to_ising
from pulse_counter_diabatic.matrix_A_et_b_vec import (
    A_direct_mat,
    b_direct_vec,
    solve_cd_torch,
)


class CounterDiabaticPulse:
    def __init__(self, seq: pulser.Sequence, config: EmulationConfig):
        (
            self.omegas_ising,  # 𝜔ᵢ 𝜎ˣᵢ
            self.mus_ising,  # 𝜇ᵢ 𝜎ʸᵢ
            self.nus_ising,  # 𝜈ᵢ 𝜎ᶻᵢ
            self.interaction_mat_ising,
        ) = from_rydberg_to_ising(seq, config)
        self.dt = config.dt
        self.n_atoms = len(seq.register.qubit_ids)

    def compute_derivatives_numerically(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiate Ising pulse parameters using 2nd-order finite
        differences.

        Uses a 3-point forward stencil at t=0, centered differences in the
        interior, and a 3-point backward stencil at t=T to preserve tensor
        shape.

        Returns:
            domegas: Time derivative of ω (σˣ coefficients), shape (T, N).
            dmus: Time derivative of μ (σʸ coefficients), shape (T, N).
            dnus: Time derivative of ν (σᶻ coefficients), shape (T, N).
        """

        def diff2(x: torch.Tensor) -> torch.Tensor:
            d0 = (-3 * x[0:1] + 4 * x[1:2] - x[2:3]) / (2 * self.dt)
            di = (x[2:] - x[:-2]) / (2 * self.dt)
            dn = (3 * x[-1:] - 4 * x[-2:-1] + x[-3:-2]) / (2 * self.dt)
            return torch.cat([d0, di, dn], dim=0)

        domegas = diff2(self.omegas_ising)  # 𝜔ᵢ'
        dmus = diff2(self.mus_ising)  # 𝜇ᵢ'
        dnus = diff2(self.nus_ising)  # 𝜈ᵢ'
        return domegas, dmus, dnus

    def solver(self):
        domegas, dmus, dnus = self.compute_derivatives_numerically()

        a_list, b_list, c_list = [], [], []
        for k in range(len(self.omegas_ising)):
            M_t = A_direct_mat(
                self.n_atoms,
                self.omegas_ising[k],
                self.mus_ising[k],
                self.nus_ising[k],
                self.interaction_mat_ising,
            )
            b_t = b_direct_vec(self.n_atoms, domegas[k], dmus[k], dnus[k])
            coeffs = solve_cd_torch(M_t, b_t, reg=1e-4)

        b_list.append(coeffs[1 : 3 * self.n_atoms : 3])  # Y
        c_list.append(coeffs[2 : 3 * self.n_atoms : 3])  # Z
        a_list.append(coeffs[0 : 3 * self.n_atoms : 3])  # X for each qubit

        a_corr = torch.stack(a_list)  # (T, n_atoms) — has grad_fn
        b_corr = torch.stack(b_list)
        c_corr = torch.stack(c_list)

        omegas_cd, mus_cd, nus_cd = (
            self.omegas_ising - a_corr,
            self.mus_ising - b_corr,
            self.nus_ising - c_corr,
        )

        return omegas_cd, mus_cd, nus_cd
