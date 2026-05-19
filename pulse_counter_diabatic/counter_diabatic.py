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

import torch.optim

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

    def _diff2(self, x: torch.Tensor) -> torch.Tensor:
        "put outside of compute_derivates_numerically to use it in cold"
        d0 = (-3 * x[0:1] + 4 * x[1:2] - x[2:3]) / (2 * self.dt)
        di = (x[2:] - x[:-2]) / (2 * self.dt)
        dn = (3 * x[-1:] - 4 * x[-2:-1] + x[-3:-2]) / (2 * self.dt)
        return 1000 * torch.cat([d0, di, dn], dim=0)

    def compute_derivatives_numerically(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._diff2(self.omegas_ising),  # 𝜔ᵢ'
            self._diff2(self.mus_ising),  # 𝜇ᵢ'
            self._diff2(self.nus_ising),  # 𝜈ᵢ'  (baseline, no COLD correction)
        )

    def solver(
        self,
        nruns: int = 10,
        cold_fourier: int = 0,
        lr: float = 1e-3,
    ) -> tuple:
        """Build the COLD-corrected Rydberg sequence.

        Args:
            nruns: Number of optimisation steps when cold_fourier > 0.
                   Ignored otherwise (single pass).
            cold_fourier: Number of Fourier modes K for the COLD bare pulse
                          (Čepaitė 2024, eq. 4.5):
                              Δν(t) = Σ_k β_k sin(2πk · t/T),  k=1..K
                          Zero at both boundaries by construction.
                          If 0, the original LCD-only behaviour is used.
            lr: Adam learning rate for β optimisation.
        """
        time_index_dim = self.omegas_ising.shape[0]

        # ── COLD: build Fourier basis ──────
        # β is the only optimised parameter
        if cold_fourier > 0:
            lam = torch.linspace(
                0.0, 1.0, time_index_dim, dtype=torch.float64
            )
            k_modes = torch.arange(
                1, cold_fourier + 1, dtype=torch.float64
            )
            sine_basis = torch.sin(
                2 * torch.pi * k_modes[None, :] * lam[:, None]
            ) 
            beta = torch.zeros(
                cold_fourier, dtype=torch.float64, requires_grad=True
            )
            optimizer = torch.optim.Adam([beta], lr=lr)
            n_steps = nruns
        else:
            n_steps = 1
        # ─────────────────────────────────────────────────────────────

        a = torch.zeros((time_index_dim, self.n_atoms), dtype=torch.float64)
        b = torch.zeros_like(a)
        c = torch.zeros_like(a)

        for step in range(n_steps):

            # ── COLD: deform ν before computing derivatives ───────────
            if cold_fourier > 0:
                optimizer.zero_grad()
                correction = sine_basis @ beta  # (T,)
                nus = self.nus_ising + correction.unsqueeze(-1)  # (T, N)
            else:
                nus = self.nus_ising
            # ─────────────────────────────────────────────────────────

            domegas = self._diff2(self.omegas_ising)
            dmus = self._diff2(self.mus_ising)
            dnus = self._diff2(nus)

            a = torch.zeros(
                (time_index_dim, self.n_atoms), dtype=torch.float64
            )
            b, c = torch.zeros_like(a), torch.zeros_like(a)
            loss = torch.tensor(0.0, dtype=torch.float64)

            for k in range(time_index_dim):
                M_t = A_direct_mat(
                    self.n_atoms,
                    self.omegas_ising[k],
                    self.mus_ising[k],
                    nus[k], 
                    self.interaction_mat_ising,
                )
                b_t = b_direct_vec(
                    self.n_atoms, domegas[k], dmus[k], dnus[k]
                )
                coeffs = solve_cd_tikhonov(M_t, b_t)

                # 2-body CD norm
                loss = loss + (coeffs[3 * self.n_atoms :] ** 2).sum()

                a[k] = coeffs[0 : 3 * self.n_atoms : 3]  # X per qubit
                b[k] = coeffs[1 : 3 * self.n_atoms : 3]  # Y per qubit
                c[k] = coeffs[2 : 3 * self.n_atoms : 3]  # Z per qubit

            # ── COLD: gradient step on β only ────────────────────────
            if cold_fourier > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([beta], max_norm=1)
                optimizer.step()
                print(
                    f"step {step:4d}  loss = {loss.item():.6e}  "
                )

                if loss.item() < 0.0001:
                    break
            else:
                print(f"LCD  loss = {loss.item():.6e}  ")
        # ─────────────────────────────────────────────────────────────

        # ── apply 1-body CD corrections ──────
        if cold_fourier > 0:
            with torch.no_grad():
                correction = sine_basis @ beta
                nus_final = (
                    self.nus_ising + correction.unsqueeze(-1)
                ).clone()
        else:
            nus_final = self.nus_ising.clone()

        with torch.no_grad():
            self.omegas_ising = self.omegas_ising + a
            self.mus_ising = self.mus_ising + b
            self.nus_ising = nus_final + c 
        # ─────────────────────────────────────────────────────────────

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