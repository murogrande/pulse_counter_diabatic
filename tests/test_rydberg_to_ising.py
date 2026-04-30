from pulse_counter_diabatic.rydberg_to_ising import (
    from_rydberg_to_ising,
    from_ising_to_rydberg,
)
from emu_mps import BitStrings
import pulser
import pytest
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
    evaluation_times = [1.0]
    observable = [BitStrings(evaluation_times=evaluation_times)]

    emu_mps_config = emu_mps.MPSConfig(
        dt=dt, interaction_matrix=interaction_matrix, observables=observable
    )

    omegas_ising, mus_ising, nus_ising, interact_matrix_ising = from_rydberg_to_ising(
        seq, emu_mps_config
    )

    expected_interaction_matrix = 0.25 * interaction_matrix
    assert torch.allclose(interact_matrix_ising, expected_interaction_matrix)

    omega = [omega_val] * int(T / dt)
    expected_omegas_ising = 0.5 * torch.stack(
        [torch.tensor(omega, dtype=torch.float64) for _ in range(num_atoms)], dim=1
    )
    assert torch.allclose(omegas_ising, expected_omegas_ising)

    nus = [delta_val / 2 - 0.25 * torch.sum(interaction_matrix[0])] * int(T / dt)
    expected_nus_ising = torch.stack(
        [torch.tensor(nus, dtype=torch.float64) for _ in range(num_atoms)], dim=1
    )
    assert torch.allclose(nus_ising, expected_nus_ising)

    expected_mus_ising = torch.zeros(int(T / dt), num_atoms, dtype=torch.float64)
    assert torch.allclose(mus_ising, expected_mus_ising)


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

    omegas_ising, mus_ising, nus_ising, interact_matrix_ising = from_rydberg_to_ising(
        seq, emu_mps_config
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
    expected_nus_ising = torch.zeros_like(nus_ising)
    for i in range(num_atoms):
        expected_nus_ising[:, i] = 0.5 * delta[:, i] - 0.25 * torch.sum(
            interaction_matrix[i]
        )

    assert torch.allclose(nus_ising, expected_nus_ising)

    expected_mus_ising = torch.zeros(int(T / dt), num_atoms, dtype=torch.float64)
    assert torch.allclose(mus_ising, expected_mus_ising)


def test_rydberg_to_ising_no_observables_warning():
    reg = pulser.Register.rectangle(1, 2, prefix="q", spacing=torch.tensor(7.0))
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulser.Pulse.ConstantPulse(100, 7.0, 5.0, 0.0), "ising_global")

    interaction_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    observable = [BitStrings(evaluation_times=[1.0])]
    config = emu_mps.MPSConfig(
        dt=10, interaction_matrix=interaction_matrix, observables=observable
    )

    with pytest.warns(UserWarning, match="initialized without any observables"):
        from_rydberg_to_ising(seq, config)


def test_ising_to_rydberg_2_atoms():
    num_atoms = 2
    duration = 10

    omegas_ising = torch.tensor([7.0] * num_atoms, dtype=torch.float64)
    omegas_ising = torch.stack([omegas_ising] * duration, dim=0)

    mus_ising = torch.tensor([5.0] * num_atoms, dtype=torch.float64)
    mus_ising = torch.stack([mus_ising] * duration, dim=0)

    nus_ising = torch.tensor([3.0] * num_atoms, dtype=torch.float64)
    nus_ising = torch.stack([nus_ising] * duration, dim=0)

    interaction_matrix_ising = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64
    )

    omegas_rydberg, mus_rydberg, nus_rydberg, interaction_matrix_rydberg = (
        from_ising_to_rydberg(
            omegas_ising, mus_ising, nus_ising, interaction_matrix_ising
        )
    )

    omegas_expected = omegas_ising
    assert torch.allclose(omegas_rydberg, omegas_expected)
    mus_expected = mus_ising
    assert torch.allclose(mus_rydberg, mus_expected)
    nus_expected = torch.zeros_like(nus_ising)
    for i in range(num_atoms):
        nus_expected[:, i] = 2 * nus_ising[:, i] + 2 * interaction_matrix_ising[i].sum()
    assert torch.allclose(nus_rydberg, nus_expected)

    interaction_matrix_expected = 4 * interaction_matrix_ising
    assert torch.allclose(interaction_matrix_rydberg, interaction_matrix_expected)


def test_ising_to_rydberg_3_atoms():
    num_atoms = 3
    duration = 10

    omegas_ising = torch.tensor([7.0] * num_atoms, dtype=torch.float64)
    omegas_ising = torch.stack([omegas_ising] * duration, dim=0)

    mus_ising = torch.tensor([5.0] * num_atoms, dtype=torch.float64)
    mus_ising = torch.stack([mus_ising] * duration, dim=0)

    nus_ising = torch.tensor([3.0] * num_atoms, dtype=torch.float64)
    nus_ising = torch.stack([nus_ising] * duration, dim=0)

    interaction_matrix_ising = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]], dtype=torch.float64
    )

    omegas_rydberg, mus_rydberg, nus_rydberg, interaction_matrix_rydberg = (
        from_ising_to_rydberg(
            omegas_ising, mus_ising, nus_ising, interaction_matrix_ising
        )
    )

    omegas_expected = omegas_ising
    assert torch.allclose(omegas_rydberg, omegas_expected)
    mus_expected = mus_ising
    assert torch.allclose(mus_rydberg, mus_expected)
    nus_expected = torch.zeros_like(nus_ising)
    for i in range(num_atoms):
        nus_expected[:, i] = 2 * nus_ising[:, i] + 2 * interaction_matrix_ising[i].sum()
    assert torch.allclose(nus_rydberg, nus_expected)

    interaction_matrix_expected = 4 * interaction_matrix_ising
    assert torch.allclose(interaction_matrix_rydberg, interaction_matrix_expected)
