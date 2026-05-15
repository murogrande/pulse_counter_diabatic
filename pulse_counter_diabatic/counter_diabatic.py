import torch
import pulser
import emu_base
from pulser.backend import EmulationConfig

from pulse_counter_diabatic.rydberg_to_ising import (
    from_rydberg_to_ising,
    from_ising_to_rydberg,
)
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
        self.seq = seq
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

        # time should be in micros, not ns
        return 1000 * domegas, 1000 * dmus, 1000 * dnus

    def solver(self, nruns: int = 10) -> tuple:
        time_index_dim = self.omegas_ising.shape[0]
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
            a = torch.zeros((time_index_dim, self.n_atoms), dtype=torch.float64)
            b, c = torch.zeros_like(a), torch.zeros_like(a)
            loss = torch.tensor(0.0, dtype=torch.float64)
            for k in range(time_index_dim):
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
                a[k] = coeffs[0 : 3 * self.n_atoms : 3]  # X per qubit
                b[k] = coeffs[1 : 3 * self.n_atoms : 3]  # Y per qubit
                c[k] = coeffs[2 : 3 * self.n_atoms : 3]  # Z per qubit

            # gradient step to reduce 2-body CD terms
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [self.omegas_ising, self.mus_ising, self.nus_ising], max_norm=1
            )
            optimizer.step()

            # direct update of 1-body with CD corrections

            print(f"step {step:4d}  loss = {loss.item():.6e}")

            if loss.item() < 0.0001:
                print(f"Early stopping at step {step} with loss {loss.item():.6f}")
                break

        domegas, dmus, dnus = self.compute_derivatives_numerically()
        a = torch.zeros((time_index_dim, self.n_atoms), dtype=torch.float64)
        b, c = torch.zeros_like(a), torch.zeros_like(a)
        for k in range(time_index_dim):
            M_t = A_direct_mat(
                self.n_atoms,
                self.omegas_ising[k],
                self.mus_ising[k],
                self.nus_ising[k],
                self.interaction_mat_ising,
            )
            b_t = b_direct_vec(self.n_atoms, domegas[k], dmus[k], dnus[k])
            coeffs = solve_cd_tikhonov(M_t, b_t)
            a[k] = coeffs[0 : 3 * self.n_atoms : 3]  # X per qubit
            b[k] = coeffs[1 : 3 * self.n_atoms : 3]  # Y per qubit
            c[k] = coeffs[2 : 3 * self.n_atoms : 3]  # Z per qubit

        with torch.no_grad():
            self.omegas_ising += a
            self.mus_ising += b
            self.nus_ising += c
        r, i, delta, interaction = from_ising_to_rydberg(
            self.omegas_ising,
            self.mus_ising,
            self.nus_ising,
            self.interaction_mat_ising,
        )

        omega = (r**2 + i**2).sqrt()
        phi = torch.atan2(i, r)
        target_times = [x * self.dt for x in range(0, omega.shape[0] + 1)]
        return emu_base.SequenceData(
            omega,
            delta,
            phi,
            lambda x: interaction,
            self.seq.register.qubit_ids,
            bad_atoms=[False] * self.n_atoms,
            lindblad_ops=[],
            state_prep_error=0.0,
            target_times=target_times,
            eigenstates=("r", "g"),
            hamiltonian_type=emu_base.HamiltonianType.Rydberg,
        )
