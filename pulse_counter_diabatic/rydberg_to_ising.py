import pulser

import emu_base
import torch
from pulser.backend import EmulationConfig


def from_rydberg_to_ising(
    seq: pulser.Sequence, config: EmulationConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a Rydberg Pulser sequence to Ising model parameters.

    Maps the Rydberg Hamiltonian (Ω/2 σˣ - δ n + U/2 nᵢnⱼ) to the Ising form
    (ω σˣ + μ σʸ + ν σᶻ + J σᶻσᶻ) via the substitution nᵢ = (1 - σᶻᵢ)/2.

    Args:
        seq: Pulser sequence encoding the Rydberg drive and detuning.
        config: Emulation config supplying the time step and device layout.

    Returns:
        omegas_ising: Half-Rabi drive, shape (T, N). Coefficients of σˣ.
        mus_ising: Zero phase tensor, shape (T, N). Coefficients of σʸ.
        nus_ising: Shifted detuning, shape (T, N). Coefficients of σᶻ.
        interact_mat_ising: Rescaled interaction matrix (U/4), shape (N, N).
    """

    cd_config = config.with_changes(observables=[])
    pulser_data = emu_base.pulser_adapter.PulserData(
        sequence=seq, config=cd_config, dt=cd_config.dt
    )
    seq0 = next(pulser_data.get_sequences())

    omegas = seq0.omega.to(dtype=torch.float64)
    deltas = seq0.delta.to(dtype=torch.float64)
    interact_mat = seq0.interaction_matrix(0.0)  # matrix is constant in time

    U_sum = interact_mat.sum(dim=1)  # (N,) — row sums, diagonal is 0
    omegas_ising = (0.5 * omegas).detach().requires_grad_(True)  # ω
    mus_ising = torch.zeros_like(omegas).requires_grad_(True)  # μ
    nus_ising = (0.5 * deltas - 0.25 * U_sum).detach().requires_grad_(True)  # ν

    interact_mat_ising = 0.25 * interact_mat
    return omegas_ising, mus_ising, nus_ising, interact_mat_ising


def from_ising_to_rydberg(
    omegas_ising: torch.Tensor,
    mus_ising: torch.Tensor,
    nus_ising: torch.Tensor,
    interact_mat_ising: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converting from Ising 𝜔ᵢ 𝜎ˣᵢ + 𝜇ᵢ  𝜎ʸᵢ +𝜈ᵢ 𝜎ᶻᵢ + Uᵢⱼ 𝜎ᶻᵢ𝜎ᶻⱼ  to Rydberg
    Hamiltonian 𝜔ᵢ 𝜎ˣᵢ + 𝜇ᵢ  𝜎ʸᵢ −𝜈ᵢ nᵢ + Uᵢⱼ nᵢ nⱼ .
    Using the substitution 𝜎ᶻᵢ = 1 - 2 nᵢ"""

    omegas_rydberg = omegas_ising

    mus_rydberg = mus_ising

    nus_rydberg = torch.zeros_like(omegas_ising)
    for i in range(omegas_ising.shape[1]):
        U_ij_sum = torch.sum(interact_mat_ising[i])
        nus_rydberg[:, i] = 2 * nus_ising[:, i] + 2 * U_ij_sum

    interact_mat_rydberg = 4 * interact_mat_ising

    return omegas_rydberg, mus_rydberg, nus_rydberg, interact_mat_rydberg


def from_rydberg_to_seq(
    omegas_rydberg: torch.Tensor,
    mus_rydberg: torch.Tensor,
    nus_rydberg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """from Rydberg Hamiltonian coefficents to pulse parameters.
    From 𝜔ᵢ 𝜎ˣᵢ + 𝜇ᵢ  𝜎ʸᵢ −𝜈ᵢ nᵢ to sequence parameters 𝛺ᵢ, 𝛿ᵢ, 𝜙ᵢ"""

    omegas_seq = 2 * torch.sqrt(omegas_rydberg**2 + mus_rydberg**2)

    phis_seq = torch.atan2(mus_rydberg, omegas_rydberg)

    deltas_seq = nus_rydberg

    return omegas_seq, deltas_seq, phis_seq
