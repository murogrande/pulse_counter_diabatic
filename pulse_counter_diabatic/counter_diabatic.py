import torch
import pulser
from pulser.backend import EmulationConfig

from pulse_counter_diabatic.rydberg_to_ising import from_rydberg_to_ising
from pulse_counter_diabatic.matrix_A_et_b_vec import (
    A_direct_mat,
    b_direct_vec,
    solve_cd_tikhonov,
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

    def solver(self, nruns: int = 10):
        optimizer = torch.optim.Adam(
            [
                {"params": self.omegas_ising, "lr": 1e-3},
                {"params": self.mus_ising, "lr": 1e-3},
                {"params": self.nus_ising, "lr": 1e-3},
            ]
        )
        for step in range(nruns):
            optimizer.zero_grad()
            domegas, dmus, dnus = self.compute_derivatives_numerically()
            a_list, b_list, c_list = [], [], []
            loss = torch.tensor(0.0, dtype=torch.float64)
            for k in range(len(self.omegas_ising)):
                M_t = A_direct_mat(
                    self.n_atoms,
                    self.omegas_ising[k],
                    self.mus_ising[k],
                    self.nus_ising[k],
                    self.interaction_mat_ising,
                )
                b_t = b_direct_vec(self.n_atoms, domegas[k], dmus[k], dnus[k])
                coeffs = solve_cd_tikhonov(M_t, b_t)

                loss = loss + (coeffs[3 * self.n_atoms :] ** 2).sum()
                a_list.append(coeffs[0 : 3 * self.n_atoms : 3])  # X per qubit
                b_list.append(coeffs[1 : 3 * self.n_atoms : 3])  # Y per qubit
                c_list.append(coeffs[2 : 3 * self.n_atoms : 3])  # Z per qubit

            # gradient step to reduce 2-body CD terms
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [self.omegas_ising, self.mus_ising, self.nus_ising], max_norm=1
            )
            optimizer.step()

            # direct update of 1-body with CD corrections
            with torch.no_grad():
                self.omegas_ising -= torch.stack(a_list).detach()
                self.mus_ising -= torch.stack(b_list).detach()
                self.nus_ising -= torch.stack(c_list).detach()

            print(f"step {step:4d}  loss = {loss.item():.6e}")

            if loss.item() < 0.0001:
                print(f"Early stopping at step {step} with loss {loss.item():.6f}")
                break

        return (
            self.omegas_ising,
            self.mus_ising,
            self.nus_ising,
            self.interaction_mat_ising,
        )
