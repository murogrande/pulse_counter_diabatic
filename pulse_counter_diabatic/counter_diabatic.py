import pulser
from pulse_counter_diabatic.rydberg_to_ising import from_rydberg_to_ising
import torch
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

    def compute_derivatives_analytical(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # finite diff the raw pulse params with 2nd-order stencils

        def diff2(x):
            d0 = (-3 * x[0:1] + 4 * x[1:2] - x[2:3]) / (2 * self.dt)
            di = (x[2:] - x[:-2]) / (2 * self.dt)
            dn = (3 * x[-1:] - 4 * x[-2:-1] + x[-3:-2]) / (2 * self.dt)
            return torch.cat([d0, di, dn], dim=0)

        domegas = diff2(self.omegas_ising)  # 𝜔ᵢ'
        dmus = diff2(self.mus_ising)  # 𝜇ᵢ'
        dnus = diff2(self.nus_ising)  # 𝜈ᵢ'
        return domegas, dmus, dnus
