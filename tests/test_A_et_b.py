from itertools import combinations, permutations

import torch

from pulse_counter_diabatic.matrix_A_et_b_vec import A_direct_mat

dtype = torch.float64


def test_A_direct_mat_2_qubits():

    num_atoms = 2

    # parameters of the pulse at specific time step
    Om_t = [torch.tensor(5.0 + 0.1 * k, dtype=dtype) for k in range(num_atoms)]
    Mu_t = [torch.tensor(3.0 + 0.1 * k, dtype=dtype) for k in range(num_atoms)]
    Nu_t = [torch.tensor(1.0 + 0.1 * k, dtype=dtype) for k in range(num_atoms)]
    U_t = {
        (i, j): torch.tensor(7.0 + 0.1 * i + 0.01 * j, dtype=dtype)
        for i, j in combinations(range(num_atoms), 2)
    }

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
    expected_A[7, 9] = Nu_t[0]
    expected_A[7, 11] = -Om_t[1]
    expected_A[8, 10] = -Mu_t[0]
    expected_A[8, 11] = Om_t[0]

    expected_A[6, 12] = -Nu_t[0]
    expected_A[6, 13] = Mu_t[0]
    expected_A[7, 12] = Nu_t[1]
    expected_A[7, 14] = -Om_t[0]
    expected_A[8, 12] = -Mu_t[1]
    expected_A[8, 13] = Om_t[1]

    expected_A[9, 10] = -Om_t[1]
    expected_A[9, 14] = Mu_t[0]
    expected_A[10, 11] = -Nu_t[0]
    expected_A[11, 12] = -Mu_t[1]
    expected_A[12, 13] = Om_t[0]
    expected_A[13, 14] = -Nu_t[1]

    expected_A = expected_A - expected_A.mT
    torch.allclose(matrix_A[3 * num_atoms, :], expected_A[3 * num_atoms, :])


# def test_b_direct():
# # derivative of Hamiltonian at a specific time step
# dOmega_t = [
#     torch.tensor(11.3 * (-1) ** k, dtype=dtype, requires_grad=True)
#     for k in range(num_atoms)
# ]
# dMu_t = [
#     torch.tensor(1.0, dtype=dtype, requires_grad=True) for _ in range(num_atoms)
# ]
# dNu_t = [
#     torch.tensor(5.1 * (-1) ** k, dtype=dtype, requires_grad=True)
#     for k in range(num_atoms)
# ]
