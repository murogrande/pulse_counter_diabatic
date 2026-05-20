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

        # Cache bare values so solver() is idempotent.
        self._omegas_bare = self.omegas_ising.clone()
        self._mus_bare = self.mus_ising.clone()
        self._nus_bare = self.nus_ising.clone()

    def _diff2(self, x: torch.Tensor) -> torch.Tensor:
        """Centered second-order finite difference w.r.t. time.

        The factor of 1000 converts ns → μs so that derivatives come out
        in (rad/μs)/μs = rad/μs² when dt is given in ns.
        """
        d0 = (-3 * x[0:1] + 4 * x[1:2] - x[2:3]) / (2 * self.dt)
        di = (x[2:] - x[:-2]) / (2 * self.dt)
        dn = (3 * x[-1:] - 4 * x[-2:-1] + x[-3:-2]) / (2 * self.dt)
        return 1000 * torch.cat([d0, di, dn], dim=0)

    def compute_derivatives_numerically(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._diff2(self.omegas_ising),
            self._diff2(self.mus_ising),
            self._diff2(self.nus_ising),
        )

    def solver(
        self,
        nruns: int = 10,
        cold_fourier: int = 0,
        lr: float = 1e-3,
        local_cold: bool = False,
        reg_beta: float = 1e-2,
    ) -> tuple:
        """Build the COLD-corrected Rydberg sequence.

        Implements Čepaitė eq. (4.4) / (6.14):
            H_COLD = H_0 + f_opt(β,λ)·σᶻ_i + λ̇·α(β,h,λ)·σʸ_i

        For a real-valued H_0 (μ_ising ≡ 0 at the bare level), the
        single-body AGP is purely σʸ. Only `b` is applied; `a` and `c`
        come out ≈ 0 by construction and are dropped.

        The σʸ correction `b` is windowed by sin²(πt/T) so that it
        vanishes at the protocol endpoints. This ensures that the final
        Pulser amplitude Ω(t) = 2·sqrt(ω² + μ²) starts and ends at zero,
        matching the bare protocol's termination and keeping the target
        Hamiltonian H(T) unchanged. The cost is a small loss of CD
        efficacy near t=0 and t=T, which is negligible when the bare
        schedule is smooth (Ω̇ ≈ 0 at the boundaries).

        Args:
            nruns: Number of optimisation steps when cold_fourier > 0.
            cold_fourier: Number of Fourier modes K. If 0, pure LCD.
            lr: Adam learning rate for β.
            local_cold: If True, β has shape (N, K). If False, shape (K,).
            reg_beta: L2 regularisation weight on β.
        """
        # Reset to bare values so solver() is idempotent.
        self.omegas_ising = self._omegas_bare.clone()
        self.mus_ising = self._mus_bare.clone()
        self.nus_ising = self._nus_bare.clone()

        time_index_dim = self.omegas_ising.shape[0]

        # ── Boundary window ────────────────────────────────────────────
        # sin²(πt/T) vanishes at t=0 and t=T with smooth derivatives.
        # Applied to the σʸ correction `b` so that μ_ising(0) =
        # μ_ising(T) = 0, ensuring Ω_Pulser(0) = Ω_Pulser(T) = 0.
        lam = torch.linspace(
            0.0, 1.0, time_index_dim, dtype=torch.float64
        )
        boundary_window = torch.sin(torch.pi * lam) ** 2   # (T,)
        # ───────────────────────────────────────────────────────────────

        # ── COLD: build Fourier basis ──────────────────────────────────
        if cold_fourier > 0:
            k_modes = torch.arange(
                1, cold_fourier + 1, dtype=torch.float64
            )
            sine_basis = torch.sin(
                2 * torch.pi * k_modes[None, :] * lam[:, None]
            )  # (T, K) — also vanishes at boundaries by construction

            if local_cold:
                beta = torch.zeros(
                    self.n_atoms,
                    cold_fourier,
                    dtype=torch.float64,
                    requires_grad=True,
                )  # (N, K)
            else:
                beta = torch.zeros(
                    cold_fourier,
                    dtype=torch.float64,
                    requires_grad=True,
                )  # (K,)

            optimizer = torch.optim.Adam([beta], lr=lr)
            n_steps = nruns
        else:
            n_steps = 1

        def build_correction():
            if not local_cold:
                return (sine_basis @ beta).unsqueeze(-1).expand(
                    -1, self.n_atoms
                )
            return sine_basis @ beta.T
        # ───────────────────────────────────────────────────────────────

        for step in range(n_steps):

            # ── COLD: deform ν before computing derivatives ────────────
            if cold_fourier > 0:
                optimizer.zero_grad()
                correction = build_correction()              # (T, N)
                nus = self.nus_ising + correction
            else:
                nus = self.nus_ising

            domegas = self._diff2(self.omegas_ising)
            dmus = self._diff2(self.mus_ising)
            dnus = self._diff2(nus)

            a = torch.zeros(
                (time_index_dim, self.n_atoms), dtype=torch.float64
            )
            b = torch.zeros_like(a)
            c = torch.zeros_like(a)
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

                # 2-body CD norm — the COLD objective.
                loss = loss + (coeffs[3 * self.n_atoms:] ** 2).sum()

                a[k] = coeffs[0 : 3 * self.n_atoms : 3]  # σˣ — dropped
                # σʸ — windowed in place so the loss reflects what is
                # actually applied to the pulse.
                b[k] = (
                    coeffs[1 : 3 * self.n_atoms : 3] * boundary_window[k]
                )
                c[k] = coeffs[2 : 3 * self.n_atoms : 3]  # σᶻ — dropped

            if cold_fourier > 0:
                loss = loss + reg_beta * (beta ** 2).sum()
                loss.backward()
                optimizer.step()
                print(
                    f"step {step:4d}  loss = {loss.item():.6e}  "
                    f"||β|| = {beta.norm().item():.3e}"
                )
                if loss.item() < 0.0001:
                    break
            else:
                print(f"LCD  loss = {loss.item():.6e}")

        # ── Build the final ν, including the COLD β-deformation ───────
        if cold_fourier > 0:
            with torch.no_grad():
                correction = build_correction()
                nus_final = (self.nus_ising + correction).clone()
        else:
            nus_final = self.nus_ising.clone()

        # ── Apply LCD: only the (already-windowed) σʸ correction ──────
        with torch.no_grad():
            # self.omegas_ising = self.omegas_ising + a   # σˣ — DROP
            self.mus_ising = self.mus_ising + b           # σʸ — KEEP (windowed)
            self.nus_ising = nus_final                    # COLD β-deformation only

        # ── Diagnostic storage ────────────────────────────────────────
        self.a_history = a.detach().clone()
        self.b_history = b.detach().clone()   # already windowed
        self.c_history = c.detach().clone()
        self.boundary_window = boundary_window.detach().clone()
        if cold_fourier > 0:
            self.beta_history = beta.detach().clone()
            self.delta_nu_cold = (
                (nus_final - self._nus_bare).detach().clone()
            )
        else:
            self.beta_history = None
            self.delta_nu_cold = None
        # ───────────────────────────────────────────────────────────────

        r, i, delta, interaction = from_ising_to_rydberg(
            self.omegas_ising,
            self.mus_ising,
            self.nus_ising,
            self.interaction_mat_ising,
        )

        omega = (r ** 2 + i ** 2).sqrt()
        phi = torch.atan2(i, r)
        target_times = [
            x * self.dt for x in range(0, omega.shape[0] + 1)
        ]
        return emu_base.SequenceData(
            omega.to(dtype=torch.complex128),
            delta.to(dtype=torch.complex128),
            phi.to(dtype=torch.complex128),
            lambda x: interaction,
            self.seq.register.qubit_ids,
            bad_atoms=[False] * self.n_atoms,
            lindblad_ops=[],
            state_prep_error=0.0,
            target_times=target_times,
            eigenstates=("r", "g"),
            hamiltonian_type=emu_base.HamiltonianType.Rydberg,
        )