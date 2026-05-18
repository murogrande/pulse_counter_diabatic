from itertools import combinations, permutations

import torch
import pulser

from pulse_counter_diabatic.matrix_A_et_b_vec import (
    A_direct_mat,
    b_direct_vec,
    solve_cd_tikhonov,
)
import emu_sv

from pulse_counter_diabatic.counter_diabatic import CounterDiabaticPulse
from utils import simulate_qutip
import numpy as np

dtype = torch.float64


def test_A_direct_mat_2_qubits():

    num_atoms = 2

    # parameters of the pulse at specific time step
    Om_t = torch.tensor([5.0 + 0.1 * k for k in range(num_atoms)], dtype=dtype)
    Mu_t = torch.tensor([3.0 + 0.1 * k for k in range(num_atoms)], dtype=dtype)
    Nu_t = torch.tensor([1.0 + 0.1 * k for k in range(num_atoms)], dtype=dtype)
    U_t = torch.zeros((num_atoms, num_atoms), dtype=dtype)
    U_t[0, 1] = torch.tensor(7.0, dtype=dtype)
    U_t[1, 0] = torch.tensor(7.0, dtype=dtype)

    matrix_A = A_direct_mat(num_atoms, Om_t, Mu_t, Nu_t, U_t)

    n_single = 3 * num_atoms  # block of single-body operators
    n_sym = 3 * len(list(combinations(range(num_atoms), 2)))  # 3 * C(n,2)
    n_asym = 3 * len(list(permutations(range(num_atoms), 2)))  # 3 * n*(n-1)
    n_total = n_single + n_sym + n_asym
    expected_A = torch.zeros(n_total, n_total, dtype=dtype)

    # block of local paulis
    expected_A[0, 1] = -Nu_t[0]
    expected_A[0, 2] = Mu_t[0]
    expected_A[1, 2] = -Om_t[0]

    expected_A[3, 4] = -Nu_t[1]
    expected_A[3, 5] = Mu_t[1]
    expected_A[4, 5] = -Om_t[1]

    # interaction matrix elements

    expected_A[0, n_single + n_sym + 2] = -U_t[0, 1]
    expected_A[1, n_single + n_sym + 1] = U_t[0, 1]

    expected_A[3, n_single + n_sym + 5] = -U_t[0, 1]
    expected_A[4, n_single + n_sym + 4] = U_t[0, 1]

    # 2 body interaction related values

    expected_A[n_single, n_single + n_sym] = -Nu_t[1]
    expected_A[n_single, n_single + n_sym + 1] = Mu_t[1]
    expected_A[n_single, n_single + n_sym + 3] = -Nu_t[0]
    expected_A[n_single, n_single + n_sym + 4] = Mu_t[0]

    expected_A[n_single + 1, n_single + n_sym] = Nu_t[0]
    expected_A[n_single + 1, n_single + n_sym + 2] = -Om_t[1]
    expected_A[n_single + 1, n_single + n_sym + 3] = Nu_t[1]
    expected_A[n_single + 1, n_single + n_sym + 5] = -Om_t[0]

    expected_A[n_single + 2, n_single + n_sym + 1] = -Mu_t[0]
    expected_A[n_single + 2, n_single + n_sym + 2] = Om_t[0]
    expected_A[n_single + 2, n_single + n_sym + 4] = -Mu_t[1]
    expected_A[n_single + 2, n_single + n_sym + 5] = Om_t[1]

    expected_A[n_single + n_sym, n_single + n_sym + 1] = -Om_t[1]
    expected_A[n_single + n_sym + 1, n_single + n_sym + 2] = -Nu_t[0]
    expected_A[n_single + n_sym + 2, n_single + n_sym + 3] = -Mu_t[1]
    expected_A[n_single + n_sym + 3, n_single + n_sym + 4] = -Om_t[0]
    expected_A[n_single + n_sym + 4, n_single + n_sym + 5] = -Nu_t[1]
    expected_A[n_single + n_sym, n_single + n_sym + 5] = Mu_t[0]

    expected_A = expected_A - expected_A.mT
    assert torch.allclose(matrix_A, expected_A)


def test_b_direct_2_qubits():

    num_atoms = 2
    n_single = 3 * num_atoms  # block of single-body operators
    n_sym = 3 * len(list(combinations(range(num_atoms), 2)))  # 3 * C(n,2)
    n_asym = 3 * len(list(permutations(range(num_atoms), 2)))  # 3 * n*(n-1)
    n_total = n_single + n_sym + n_asym

    # derivative of Hamiltonian at a specific time step
    dOmega_t = torch.tensor([11.3 * (-1) ** k for k in range(num_atoms)], dtype=dtype)
    dMu_t = torch.tensor([1.0 for _ in range(num_atoms)], dtype=dtype)
    dNu_t = torch.tensor([5.1 * (-1) ** k for k in range(num_atoms)], dtype=dtype)

    b_vector = b_direct_vec(num_atoms, dOmega_t, dMu_t, dNu_t)

    expected_b = torch.zeros(n_total, dtype=dtype)

    expected_b[0] = -dOmega_t[0] / 2
    expected_b[1] = -dMu_t[0] / 2
    expected_b[2] = -dNu_t[0] / 2
    expected_b[3] = -dOmega_t[1] / 2
    expected_b[4] = -dMu_t[1] / 2
    expected_b[5] = -dNu_t[1] / 2

    assert torch.allclose(b_vector, expected_b)


def test_A_direct_mat_3_qubits():

    num_atoms = 3

    # parameters of the pulse at specific time step
    Om_t = torch.tensor([5.0 + 0.1 * k for k in range(num_atoms)], dtype=dtype)
    Mu_t = torch.tensor([3.0 + 0.1 * k for k in range(num_atoms)], dtype=dtype)
    Nu_t = torch.tensor([1.0 + 0.1 * k for k in range(num_atoms)], dtype=dtype)
    U_t = torch.zeros((num_atoms, num_atoms), dtype=dtype)
    for i, j in combinations(range(num_atoms), 2):
        value = torch.tensor(7.0 + 0.1 * i + 0.01 * j, dtype=dtype)
        U_t[i, j] = value
        U_t[j, i] = value

    matrix_A = A_direct_mat(num_atoms, Om_t, Mu_t, Nu_t, U_t)

    n_single = 3 * num_atoms  # block of single-body operators
    n_sym = 3 * len(list(combinations(range(num_atoms), 2)))  # 3 * C(n,2)
    n_asym = 3 * len(list(permutations(range(num_atoms), 2)))  # 3 * n*(n-1)
    n_total = n_single + n_sym + n_asym
    expected_A = torch.zeros(n_total, n_total, dtype=dtype)

    # block of local paulis
    expected_A[0, 1] = -Nu_t[0]
    expected_A[0, 2] = Mu_t[0]
    expected_A[1, 2] = -Om_t[0]

    expected_A[3, 4] = -Nu_t[1]
    expected_A[3, 5] = Mu_t[1]
    expected_A[4, 5] = -Om_t[1]

    expected_A[6, 7] = -Nu_t[2]
    expected_A[6, 8] = Mu_t[2]
    expected_A[7, 8] = -Om_t[2]

    # interaction matrix elements

    expected_A[0, n_single + n_sym + 2] = -U_t[0, 1]
    expected_A[1, n_single + n_sym + 1] = U_t[0, 1]

    expected_A[0, n_single + n_sym + 8] = -U_t[0, 2]
    expected_A[1, n_single + n_sym + 7] = U_t[0, 2]

    expected_A[3, n_single + n_sym + 5] = -U_t[0, 1]
    expected_A[4, n_single + n_sym + 4] = U_t[0, 1]

    expected_A[6, n_single + n_sym + 11] = -U_t[0, 2]
    expected_A[7, n_single + n_sym + 10] = U_t[0, 2]

    expected_A[3, n_single + n_sym + 14] = -U_t[1, 2]
    expected_A[4, n_single + n_sym + 13] = U_t[1, 2]
    expected_A[6, n_single + n_sym + 17] = -U_t[1, 2]
    expected_A[7, n_single + n_sym + 16] = U_t[1, 2]

    # 2 body interaction -  symmetric

    expected_A[n_single, n_single + n_sym] = -Nu_t[1]
    expected_A[n_single, n_single + n_sym + 1] = Mu_t[1]
    expected_A[n_single + 1, n_single + n_sym] = Nu_t[0]
    expected_A[n_single + 1, n_single + n_sym + 2] = -Om_t[1]
    expected_A[n_single + 2, n_single + n_sym + 1] = -Mu_t[0]
    expected_A[n_single + 2, n_single + n_sym + 2] = Om_t[0]

    expected_A[n_single, n_single + n_sym + 3] = -Nu_t[0]
    expected_A[n_single, n_single + n_sym + 4] = Mu_t[0]
    expected_A[n_single + 1, n_single + n_sym + 3] = Nu_t[1]
    expected_A[n_single + 1, n_single + n_sym + 5] = -Om_t[0]
    expected_A[n_single + 2, n_single + n_sym + 4] = -Mu_t[1]
    expected_A[n_single + 2, n_single + n_sym + 5] = Om_t[1]

    expected_A[n_single + 3, n_single + n_sym + 6] = -Nu_t[2]
    expected_A[n_single + 3, n_single + n_sym + 7] = Mu_t[2]
    expected_A[n_single + 4, n_single + n_sym + 6] = Nu_t[0]
    expected_A[n_single + 4, n_single + n_sym + 8] = -Om_t[2]
    expected_A[n_single + 5, n_single + n_sym + 7] = -Mu_t[0]
    expected_A[n_single + 5, n_single + n_sym + 8] = Om_t[0]

    expected_A[n_single + 3, n_single + n_sym + 9] = -Nu_t[0]
    expected_A[n_single + 3, n_single + n_sym + 10] = Mu_t[0]
    expected_A[n_single + 4, n_single + n_sym + 9] = Nu_t[2]
    expected_A[n_single + 4, n_single + n_sym + 11] = -Om_t[0]
    expected_A[n_single + 5, n_single + n_sym + 10] = -Mu_t[2]
    expected_A[n_single + 5, n_single + n_sym + 11] = Om_t[2]

    expected_A[n_single + 6, n_single + n_sym + 12] = -Nu_t[2]
    expected_A[n_single + 6, n_single + n_sym + 13] = Mu_t[2]
    expected_A[n_single + 7, n_single + n_sym + 12] = Nu_t[1]
    expected_A[n_single + 7, n_single + n_sym + 14] = -Om_t[2]
    expected_A[n_single + 8, n_single + n_sym + 13] = -Mu_t[1]
    expected_A[n_single + 8, n_single + n_sym + 14] = Om_t[1]

    expected_A[n_single + 6, n_single + n_sym + 15] = -Nu_t[1]
    expected_A[n_single + 6, n_single + n_sym + 16] = Mu_t[1]
    expected_A[n_single + 7, n_single + n_sym + 15] = Nu_t[2]
    expected_A[n_single + 7, n_single + n_sym + 17] = -Om_t[1]
    expected_A[n_single + 8, n_single + n_sym + 16] = -Mu_t[2]
    expected_A[n_single + 8, n_single + n_sym + 17] = Om_t[2]

    # 2 body interacions -  asymmetrical
    expected_A[n_single + n_sym, n_single + n_sym + 1] = -Om_t[1]
    expected_A[n_single + n_sym + 1, n_single + n_sym + 2] = -Nu_t[0]
    expected_A[n_single + n_sym + 2, n_single + n_sym + 3] = -Mu_t[1]
    expected_A[n_single + n_sym + 3, n_single + n_sym + 4] = -Om_t[0]
    expected_A[n_single + n_sym + 4, n_single + n_sym + 5] = -Nu_t[1]
    expected_A[n_single + n_sym, n_single + n_sym + 5] = Mu_t[0]

    expected_A[n_single + n_sym + 6, n_single + n_sym + 7] = -Om_t[2]
    expected_A[n_single + n_sym + 7, n_single + n_sym + 8] = -Nu_t[0]
    expected_A[n_single + n_sym + 8, n_single + n_sym + 9] = -Mu_t[2]
    expected_A[n_single + n_sym + 9, n_single + n_sym + 10] = -Om_t[0]
    expected_A[n_single + n_sym + 10, n_single + n_sym + 11] = -Nu_t[2]
    expected_A[n_single + n_sym + 6, n_single + n_sym + 11] = Mu_t[0]

    expected_A[n_single + n_sym + 12, n_single + n_sym + 13] = -Om_t[2]
    expected_A[n_single + n_sym + 13, n_single + n_sym + 14] = -Nu_t[1]
    expected_A[n_single + n_sym + 14, n_single + n_sym + 15] = -Mu_t[2]
    expected_A[n_single + n_sym + 15, n_single + n_sym + 16] = -Om_t[1]
    expected_A[n_single + n_sym + 16, n_single + n_sym + 17] = -Nu_t[2]
    expected_A[n_single + n_sym + 12, n_single + n_sym + 17] = Mu_t[1]

    expected_A = expected_A - expected_A.mT
    assert torch.allclose(matrix_A, expected_A)


def test_b_direct_3_qubits():

    num_atoms = 3
    n_single = 3 * num_atoms  # block of single-body operators
    n_sym = 3 * len(list(combinations(range(num_atoms), 2)))  # 3 * C(n,2)
    n_asym = 3 * len(list(permutations(range(num_atoms), 2)))  # 3 * n*(n-1)
    n_total = n_single + n_sym + n_asym

    # derivative of Hamiltonian at a specific time step
    dOmega_t = torch.tensor([11.3 * (-1) ** k for k in range(num_atoms)], dtype=dtype)
    dMu_t = torch.tensor([1.0 for _ in range(num_atoms)], dtype=dtype)
    dNu_t = torch.tensor([5.1 * (-1) ** k for k in range(num_atoms)], dtype=dtype)

    b_vector = b_direct_vec(num_atoms, dOmega_t, dMu_t, dNu_t)

    expected_b = torch.zeros(n_total, dtype=dtype)

    expected_b[0] = -dOmega_t[0] / 2
    expected_b[1] = -dMu_t[0] / 2
    expected_b[2] = -dNu_t[0] / 2
    expected_b[3] = -dOmega_t[1] / 2
    expected_b[4] = -dMu_t[1] / 2
    expected_b[5] = -dNu_t[1] / 2
    expected_b[6] = -dOmega_t[2] / 2
    expected_b[7] = -dMu_t[2] / 2
    expected_b[8] = -dNu_t[2] / 2

    assert torch.allclose(b_vector, expected_b)


def test_2_qubit_sequence():
    n_qubits = 2

    reg = pulser.Register.rectangle(1, n_qubits, prefix="q", spacing=torch.tensor(8))

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
    domegas, dmus, dnus = counter_diabatic_pulse.compute_derivatives_numerically()

    time_index_dim = counter_diabatic_pulse.omegas_ising.shape[0]
    coeffs = torch.zeros(
        (
            time_index_dim,
            3 * n_qubits
            + 3 * n_qubits * (n_qubits - 1) // 2
            + 3 * n_qubits * (n_qubits - 1),
        ),
        dtype=torch.float64,
    )
    for k in range(time_index_dim):
        M_t = A_direct_mat(
            counter_diabatic_pulse.n_atoms,
            counter_diabatic_pulse.omegas_ising[k],
            counter_diabatic_pulse.mus_ising[k],
            counter_diabatic_pulse.nus_ising[k],
            counter_diabatic_pulse.interaction_mat_ising,
        )
        b_t = b_direct_vec(n_qubits, domegas[k], dmus[k], dnus[k])
        coeffs[k] = solve_cd_tikhonov(M_t, b_t)
    sol = simulate_qutip(
        counter_diabatic_pulse.omegas_ising,
        counter_diabatic_pulse.mus_ising,
        counter_diabatic_pulse.nus_ising,
        counter_diabatic_pulse.interaction_mat_ising,
        coeffs,
        [x + 0.5 for x in range(time_index_dim)],
    )

    afm = np.array(
        [
            np.abs(x.full()[1].item()) ** 2 + np.abs(x.full()[2].item()) ** 2
            for x in sol.states
        ]
    )

    assert afm[-1] > 1 - 1e-8  # for 2 qubits, the counter term is exact


def test_3_qubit_sequence():
    n_qubits = 3

    reg = pulser.Register.rectangle(1, n_qubits, prefix="q", spacing=torch.tensor(8))

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
    domegas, dmus, dnus = counter_diabatic_pulse.compute_derivatives_numerically()

    time_index_dim = counter_diabatic_pulse.omegas_ising.shape[0]
    coeffs = torch.zeros(
        (
            time_index_dim,
            3 * n_qubits
            + 3 * n_qubits * (n_qubits - 1) // 2
            + 3 * n_qubits * (n_qubits - 1),
        ),
        dtype=torch.float64,
    )
    for k in range(time_index_dim):
        M_t = A_direct_mat(
            counter_diabatic_pulse.n_atoms,
            counter_diabatic_pulse.omegas_ising[k],
            counter_diabatic_pulse.mus_ising[k],
            counter_diabatic_pulse.nus_ising[k],
            counter_diabatic_pulse.interaction_mat_ising,
        )
        b_t = b_direct_vec(n_qubits, domegas[k], dmus[k], dnus[k])
        coeffs[k] = solve_cd_tikhonov(M_t, b_t)
    sol = simulate_qutip(
        counter_diabatic_pulse.omegas_ising,
        counter_diabatic_pulse.mus_ising,
        counter_diabatic_pulse.nus_ising,
        counter_diabatic_pulse.interaction_mat_ising,
        coeffs,
        [x + 0.5 for x in range(time_index_dim)],
    )

    afm = np.array([np.abs(x.full()[5].item()) ** 2 for x in sol.states])

    assert afm[-1] > 0.93  # for 3 qubits, the counter term is not exact
