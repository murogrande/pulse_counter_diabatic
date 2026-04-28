from itertools import combinations, permutations

import torch

from pulse_counter_diabatic.matrix_A_et_b_vec import A_direct_mat, b_direct_vec

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

    expected_A[0, 11] = -U_t[0, 1]
    expected_A[1, 10] = U_t[0, 1]

    expected_A[3, 14] = -U_t[0, 1]
    expected_A[4, 13] = U_t[0, 1]

    # 2 body interaction related values

    expected_A[6, 9] = -Nu_t[1]
    expected_A[6, 10] = Mu_t[1]
    expected_A[6, 12] = -Nu_t[0]
    expected_A[6, 13] = Mu_t[0]

    expected_A[7, 9] = Nu_t[0]
    expected_A[7, 11] = -Om_t[1]
    expected_A[7, 12] = Nu_t[1]
    expected_A[7, 14] = -Om_t[0]

    expected_A[8, 10] = -Mu_t[0]
    expected_A[8, 11] = Om_t[0]
    expected_A[8, 13] = -Mu_t[1]
    expected_A[8, 14] = Om_t[1]

    expected_A[9, 10] = -Om_t[1]
    expected_A[10, 11] = -Nu_t[0]
    expected_A[11, 12] = -Mu_t[1]
    expected_A[12, 13] = -Om_t[0]
    expected_A[13, 14] = -Nu_t[1]
    expected_A[9, 14] = Mu_t[0]

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

    expected_A[0, 20] = -U_t[0, 1]
    expected_A[1, 19] = U_t[0, 1]

    expected_A[0, 26] = -U_t[0, 2]
    expected_A[1, 25] = U_t[0, 2]

    expected_A[3, 23] = -U_t[0, 1]
    expected_A[4, 22] = U_t[0, 1]

    expected_A[6, 29] = -U_t[0, 2]
    expected_A[7, 28] = U_t[0, 2]

    expected_A[3, 32] = -U_t[1, 2]
    expected_A[4, 31] = U_t[1, 2]
    expected_A[6, 35] = -U_t[1, 2]
    expected_A[7, 34] = U_t[1, 2]

    # 2 body interaction -  symmetric

    expected_A[9, 18] = -Nu_t[1]
    expected_A[9, 19] = Mu_t[1]
    expected_A[10, 18] = Nu_t[0]
    expected_A[10, 20] = -Om_t[1]
    expected_A[11, 19] = -Mu_t[0]
    expected_A[11, 20] = Om_t[0]

    expected_A[9, 21] = -Nu_t[0]
    expected_A[9, 22] = Mu_t[0]
    expected_A[10, 21] = Nu_t[1]
    expected_A[10, 23] = -Om_t[0]
    expected_A[11, 22] = -Mu_t[1]
    expected_A[11, 23] = Om_t[1]

    expected_A[12, 24] = -Nu_t[2]
    expected_A[12, 25] = Mu_t[2]
    expected_A[13, 24] = Nu_t[0]
    expected_A[13, 26] = -Om_t[2]
    expected_A[14, 25] = -Mu_t[0]
    expected_A[14, 26] = Om_t[0]

    expected_A[12, 27] = -Nu_t[0]
    expected_A[12, 28] = Mu_t[0]
    expected_A[13, 27] = Nu_t[2]
    expected_A[13, 29] = -Om_t[0]
    expected_A[14, 28] = -Mu_t[2]
    expected_A[14, 29] = Om_t[2]

    expected_A[15, 30] = -Nu_t[2]
    expected_A[15, 31] = Mu_t[2]
    expected_A[16, 30] = Nu_t[1]
    expected_A[16, 32] = -Om_t[2]
    expected_A[17, 31] = -Mu_t[1]
    expected_A[17, 32] = Om_t[1]

    expected_A[15, 33] = -Nu_t[1]
    expected_A[15, 34] = Mu_t[1]
    expected_A[16, 33] = Nu_t[2]
    expected_A[16, 35] = -Om_t[1]
    expected_A[17, 34] = -Mu_t[2]
    expected_A[17, 35] = Om_t[2]

    # 2 body interacions -  asymmetrical
    expected_A[18, 19] = -Om_t[1]
    expected_A[19, 20] = -Nu_t[0]
    expected_A[20, 21] = -Mu_t[1]
    expected_A[21, 22] = -Om_t[0]
    expected_A[22, 23] = -Nu_t[1]
    expected_A[18, 23] = Mu_t[0]

    expected_A[24, 25] = -Om_t[2]
    expected_A[25, 26] = -Nu_t[0]
    expected_A[26, 27] = -Mu_t[2]
    expected_A[27, 28] = -Om_t[0]
    expected_A[28, 29] = -Nu_t[2]
    expected_A[24, 29] = Mu_t[0]

    expected_A[30, 31] = -Om_t[2]
    expected_A[31, 32] = -Nu_t[1]
    expected_A[32, 33] = -Mu_t[2]
    expected_A[33, 34] = -Om_t[1]
    expected_A[34, 35] = -Nu_t[2]
    expected_A[30, 35] = Mu_t[1]

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
