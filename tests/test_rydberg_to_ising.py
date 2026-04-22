from pulse_counter_diabatic.rydberg_to_ising import from_rydberg_to_ising
import pulser
import torch
import emu_mps


def test_rydberg_to_ising_2_atoms():
    num_atoms = 2

    reg = pulser.Register.rectangle(1, num_atoms, prefix="q", spacing=torch.tensor(7.0))

    T = 100
    omega_val = 7.0
    delta_val = 5.0
    constant_pulse = pulser.Pulse.ConstantPulse(T, omega_val, delta_val, 0.0)
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(constant_pulse, "ising_global")

    dt = 10
    interaction_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    emu_mps_config = emu_mps.MPSConfig(dt=dt, interaction_matrix=interaction_matrix)

    omegas_ising, deltas_ising, phis_ising, interact_matrix_ising = (
        from_rydberg_to_ising(seq, emu_mps_config)
    )

    expected_interaction_matrix = 0.25 * interaction_matrix
    assert torch.allclose(interact_matrix_ising, expected_interaction_matrix)

    omega = [omega_val] * int(T / dt)
    expected_omegas_ising = 0.5 * torch.stack(
        [torch.tensor(omega, dtype=torch.float64) for _ in range(num_atoms)], dim=1
    )
    assert torch.allclose(omegas_ising, expected_omegas_ising)

    delta = [delta_val / 2 - 0.25 * torch.sum(interaction_matrix[0])] * int(T / dt)
    expected_deltas_ising = torch.stack(
        [torch.tensor(delta, dtype=torch.float64) for _ in range(num_atoms)], dim=1
    )
    assert torch.allclose(deltas_ising, expected_deltas_ising)

    expected_phis_ising = torch.zeros(int(T / dt), num_atoms, dtype=torch.float64)
    assert torch.allclose(phis_ising, expected_phis_ising)


def test_rydberg_to_ising_3_atoms():
    num_atoms = 3

    reg = pulser.Register.rectangle(1, num_atoms, prefix="q", spacing=torch.tensor(7.0))

    T = 100
    omega_val = 7.0
    delta_val = 5.0
    constant_pulse = pulser.Pulse.ConstantPulse(T, omega_val, delta_val, 0.0)
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(constant_pulse, "ising_global")

    dt = 10
    interaction_matrix = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]], dtype=torch.float64
    )
    emu_mps_config = emu_mps.MPSConfig(dt=dt, interaction_matrix=interaction_matrix)

    omegas_ising, deltas_ising, phis_ising, interact_matrix_ising = (
        from_rydberg_to_ising(seq, emu_mps_config)
    )

    expected_interaction_matrix = 0.25 * interaction_matrix
    assert torch.allclose(interact_matrix_ising, expected_interaction_matrix)

    omega = [omega_val] * int(T / dt)
    expected_omegas_ising = 0.5 * torch.stack(
        [torch.tensor(omega, dtype=torch.float64) for _ in range(num_atoms)], dim=1
    )
    assert torch.allclose(omegas_ising, expected_omegas_ising)

    delta = torch.tensor([delta_val] * int(T / dt))
    delta = torch.stack([delta] * num_atoms, dim=1)
    expected_deltas_ising = torch.zeros_like(deltas_ising)
    for i in range(num_atoms):
        expected_deltas_ising[:, i] = 0.5 * delta[:, i] - 0.25 * torch.sum(
            interaction_matrix[i]
        )

    assert torch.allclose(deltas_ising, expected_deltas_ising)

    expected_phis_ising = torch.zeros(int(T / dt), num_atoms, dtype=torch.float64)
    assert torch.allclose(phis_ising, expected_phis_ising)
