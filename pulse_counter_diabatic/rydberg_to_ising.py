import pulser

import emu_base
import torch
from pulser.backend import EmulationConfig


def from_rydberg_to_ising(
    seq: pulser.Sequence, config: EmulationConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    dt = config.dt
    pulser_data = emu_base.pulser_adapter.PulserData(sequence=seq, config=config, dt=dt)
    sequence_data = pulser_data.get_sequences()
    omegas = [
        i.omega.to(dtype=torch.float64).requires_grad_(True) for i in sequence_data
    ]
    sequence_data = pulser_data.get_sequences()
    deltas = [
        i.delta.to(dtype=torch.float64).requires_grad_(True) for i in sequence_data
    ]
    sequence_data = pulser_data.get_sequences()

    interact_full = [i.interaction_matrix.full_matrix for i in sequence_data]

    omegas = omegas[0]  # only torch tensors for optimization loop
    deltas = deltas[0]
    # phis = phis[0] # only real hamiltonians
    interact_mat = interact_full[0]  # (n_atoms, n_atoms), time-independent

    omegas_ising = 0.5 * omegas
    deltas_ising = torch.zeros_like(deltas)
    for i in range(deltas.shape[1]):
        U_ij_sum = torch.sum(interact_mat[i])
        deltas_ising[:, i] = 0.5 * deltas[:, i] - 0.25 * U_ij_sum
    phis_ising = torch.zeros_like(omegas)
    interact_mat_ising = 0.25 * interact_mat
    return omegas_ising, deltas_ising, phis_ising, interact_mat_ising
