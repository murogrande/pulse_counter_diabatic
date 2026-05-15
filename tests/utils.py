import qutip
import numpy as np
from itertools import combinations


def single_qubit_gate(n_qubits, i, op):
    ops = [qutip.operators.qeye(2)] * n_qubits
    ops[i] = op
    return qutip.tensor(ops)


def two_qubit_gate(n_qubits, i, j, op1, op2):
    ops = [qutip.operators.qeye(2)] * n_qubits
    ops[i] = op1
    ops[j] = op2
    return qutip.tensor(ops)


def simulate_qutip(omega, mu, nu, interaction, counter_terms, times):
    np_times = np.array(times) / 1000
    n_qubits = omega.shape[-1]
    zero = qutip.states.basis(2, 0)

    x, y, z = (
        qutip.operators.sigmax(),
        qutip.operators.sigmay(),
        qutip.operators.sigmaz(),
    )
    op_list = []
    for i in range(n_qubits):
        op_list.append(
            [
                single_qubit_gate(n_qubits, i, x),
                (omega[:, i] - counter_terms[:, 3 * i]).detach().numpy(),
            ]
        )
        op_list.append(
            [
                single_qubit_gate(n_qubits, i, y),
                (mu[:, i] - counter_terms[:, 3 * i + 1]).detach().numpy(),
            ]
        )
        op_list.append(
            [
                single_qubit_gate(n_qubits, i, z),
                (nu[:, i] - counter_terms[:, 3 * i + 2]).detach().numpy(),
            ]
        )
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            op_list.append(
                [
                    two_qubit_gate(n_qubits, i, j, z, z),
                    np.array([interaction[i, j]] * len(times)),
                ]
            )

    n_single = 3 * n_qubits
    n_sym = 3 * n_qubits * (n_qubits - 1) // 2

    for i, c in enumerate(combinations(range(n_qubits), 2)):
        op_list.append(
            [
                two_qubit_gate(n_qubits, c[0], c[1], x, x),
                -counter_terms[:, n_single + 3 * i].detach().numpy(),
            ]
        )
        op_list.append(
            [
                two_qubit_gate(n_qubits, c[0], c[1], y, y),
                -counter_terms[:, n_single + 3 * i + 1].detach().numpy(),
            ]
        )
        op_list.append(
            [
                two_qubit_gate(n_qubits, c[0], c[1], y, y),
                -counter_terms[:, n_single + 3 * i + 2].detach().numpy(),
            ]
        )

        op_list.append(
            [
                two_qubit_gate(n_qubits, c[0], c[1], x, y),
                -counter_terms[:, n_single + n_sym + 3 * i].detach().numpy(),
            ]
        )
        op_list.append(
            [
                two_qubit_gate(n_qubits, c[0], c[1], x, z),
                -counter_terms[:, n_single + n_sym + 3 * i + 1].detach().numpy(),
            ]
        )
        op_list.append(
            [
                two_qubit_gate(n_qubits, c[0], c[1], y, z),
                -counter_terms[:, n_single + n_sym + 3 * i + 2].detach().numpy(),
            ]
        )

        op_list.append(
            [
                two_qubit_gate(n_qubits, c[0], c[1], y, x),
                -counter_terms[:, n_single + n_sym + 3 * i + 3].detach().numpy(),
            ]
        )
        op_list.append(
            [
                two_qubit_gate(n_qubits, c[0], c[1], z, x),
                -counter_terms[:, n_single + n_sym + 3 * i + 4].detach().numpy(),
            ]
        )
        op_list.append(
            [
                two_qubit_gate(n_qubits, c[0], c[1], z, y),
                -counter_terms[:, n_single + n_sym + 3 * i + 5].detach().numpy(),
            ]
        )

    h = qutip.QobjEvo(op_list, tlist=np_times)

    return qutip.sesolve(
        h, qutip.tensor([zero] * n_qubits), np.concatenate([[0], np_times, [1]])
    )
