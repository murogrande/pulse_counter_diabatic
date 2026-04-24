import pulser
import torch
import emu_mps

from pulse_counter_diabatic.counter_diabatic import CounterDiabaticPulse


def test_derivative():
    """Derivative of the adiabatic pulse with respect to omega and mu. The pulses are
    defined as lines"""
    num_atoms = 2

    reg = pulser.Register.rectangle(1, num_atoms, prefix="q", spacing=torch.tensor(7.0))

    T = 100
    omega_val = 7.0
    delta_val = 10.0

    adiabatic_pulse = pulser.Pulse(
        pulser.InterpolatedWaveform(T, [0, omega_val]),
        pulser.InterpolatedWaveform(T, [-delta_val, 0, delta_val]),
        0,
    )
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(adiabatic_pulse, "ising_global")

    dt = 10
    interaction_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    emu_mps_config = emu_mps.MPSConfig(dt=dt, interaction_matrix=interaction_matrix)

    counter_diabatic_pulse = CounterDiabaticPulse(seq, emu_mps_config)

    domega, dmu, dnu = counter_diabatic_pulse.compute_derivatives_analytical()

    omega_expected = torch.tensor(
        [omega_val / 2 / T] * int(T / dt), dtype=torch.float64
    )
    omega_expected = omega_expected.unsqueeze(1).repeat(1, num_atoms)
    assert torch.allclose(domega, omega_expected, atol=1e-2)

    nu_expected = torch.tensor(
        [(2 * delta_val / 2 - 0.25 * interaction_matrix[0].sum()) / T] * int(T / dt),
        dtype=torch.float64,
    )

    nu_expected = nu_expected.unsqueeze(1).repeat(1, num_atoms)
    assert torch.allclose(dnu, nu_expected, atol=1e-2)

    mu_expected = torch.zeros(int(T / dt), num_atoms, dtype=torch.float64)
    assert torch.allclose(dmu, mu_expected)
