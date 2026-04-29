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
        nus_ising: Zero phase tensor, shape (T, N). Coefficients of σʸ.
        mus_ising: Shifted detuning, shape (T, N). Coefficients of σᶻ.
        interact_mat_ising: Rescaled interaction matrix (U/4), shape (N, N).
    """

    cd_config = config.with_changes(observables=[])
    pulser_data = emu_base.pulser_adapter.PulserData(
        sequence=seq, config=cd_config, dt=cd_config.dt
    )
    seq0 = next(pulser_data.get_sequences())

    omegas = seq0.omega.to(dtype=torch.float64).requires_grad_(True)
    deltas = seq0.delta.to(dtype=torch.float64).requires_grad_(True)
    interact_mat = seq0.interaction_matrix(0.0)  # matrix is constant in time

    omegas_ising = 0.5 * omegas  # ω
    mus_ising = torch.zeros_like(omegas)  # μ
    nus_ising = torch.zeros_like(deltas)  # ν
    for i in range(deltas.shape[1]):
        U_ij_sum = torch.sum(interact_mat[i])
        nus_ising[:, i] = 0.5 * deltas[:, i] - 0.25 * U_ij_sum

    interact_mat_ising = 0.25 * interact_mat
    return omegas_ising, mus_ising, nus_ising, interact_mat_ising


def from_ising_to_rydberg(
    omegas_ising: torch.Tensor,
    mus_ising: torch.Tensor,
    nus_ising: torch.Tensor,
    interact_mat_ising: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converting from Ising 𝜔ᵢ 𝜎ˣᵢ + 𝜇ᵢ  𝜎ʸᵢ +𝜈ᵢ 𝜎ᶻᵢ + Uᵢⱼ to Rydberg
    Hamiltonian. Using the substitution 𝜎ᶻᵢ = 1 - 2 nᵢ"""

    omegas_rydberg = omegas_ising

    mus_rydberg = mus_ising

    nus_rydberg = torch.zeros_like(omegas_ising)
    for i in range(omegas_ising.shape[1]):
        U_ij_sum = torch.sum(interact_mat_ising[i])
        nus_rydberg[:, i] = 2 * nus_ising[:, i] + 2 * U_ij_sum

    interact_mat_rydberg = 4 * interact_mat_ising

    return omegas_rydberg, mus_rydberg, nus_rydberg, interact_mat_rydberg
