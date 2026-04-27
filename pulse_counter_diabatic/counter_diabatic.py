import torch
import pulser
from pulse_counter_diabatic.rydberg_to_ising import from_rydberg_to_ising
from pulser.backend import EmulationConfig


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

    def compute_derivatives_analytical(
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

        def diff2(x):
            d0 = (-3 * x[0:1] + 4 * x[1:2] - x[2:3]) / (2 * self.dt)
            di = (x[2:] - x[:-2]) / (2 * self.dt)
            dn = (3 * x[-1:] - 4 * x[-2:-1] + x[-3:-2]) / (2 * self.dt)
            return torch.cat([d0, di, dn], dim=0)

        domegas = diff2(self.omegas_ising)  # 𝜔ᵢ'
        dmus = diff2(self.mus_ising)  # 𝜇ᵢ'
        dnus = diff2(self.nus_ising)  # 𝜈ᵢ'
        return domegas, dmus, dnus
