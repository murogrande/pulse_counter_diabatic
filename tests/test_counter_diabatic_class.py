import pulser
import torch
import emu_sv
import numpy as np

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
        pulser.InterpolatedWaveform(T, [-delta_val, delta_val]),
        0,
    )
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(adiabatic_pulse, "ising_global")

    dt = 10
    interaction_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    emu_sv_config = emu_sv.SVConfig(dt=dt, interaction_matrix=interaction_matrix)

    counter_diabatic_pulse = CounterDiabaticPulse(seq, emu_sv_config)

    domega, dmu, dnu = counter_diabatic_pulse.compute_derivatives_numerically()

    omega_expected = (
        torch.tensor([omega_val / 2 / (T - 1)] * int(T / dt), dtype=torch.float64)
        * 1000
    )  # convert to microsec
    omega_expected = omega_expected.unsqueeze(1).repeat(1, num_atoms)
    assert torch.allclose(domega, omega_expected)

    nu_expected = (
        torch.tensor(
            [(2 * delta_val / 2) / (T - 1)] * int(T / dt),
            dtype=torch.float64,
        )
        * 1000
    )  # convert to microsec

    nu_expected = nu_expected.unsqueeze(1).repeat(1, num_atoms)
    assert torch.allclose(dnu, nu_expected)

    mu_expected = torch.zeros(int(T / dt), num_atoms, dtype=torch.float64)
    assert torch.allclose(dmu, mu_expected)


def test_no_interaction():
    n_qubits = 3

    reg = pulser.Register.rectangle(1, n_qubits, prefix="q", spacing=torch.tensor(1e6))

    T = 1000

    adiabatic_pulse = pulser.Pulse(
        pulser.InterpolatedWaveform(
            T, 2 * torch.sin(torch.arange(0, torch.pi, torch.pi / T))
        ),
        pulser.InterpolatedWaveform(
            T, -2 * torch.cos(torch.arange(0, torch.pi, torch.pi / T))
        ),
        0,
    )
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(adiabatic_pulse, "ising_global")

    dt = 1
    config = emu_sv.SVConfig(dt=dt, observables=[emu_sv.Occupation()])

    counter_diabatic_pulse = CounterDiabaticPulse(seq, config)
    solution = counter_diabatic_pulse.solver()
    config = emu_sv.SVConfig(
        dt=dt,
        observables=[
            emu_sv.Occupation(evaluation_times=np.array(solution.target_times) / 1000)
        ],
    )

    results2 = emu_sv.SVBackend._run_from_sequence_data(solution, config)
    occupation2 = results2.occupation[-1]
    for x in occupation2:
        assert x > 0.97
