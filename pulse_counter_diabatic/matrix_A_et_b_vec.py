from itertools import combinations, permutations

import torch

dtype = torch.float64


def A_direct_mat(
    n_atoms: int,
    Omega_t: torch.Tensor,
    Mu_t: torch.Tensor,
    Nu_t: torch.Tensor,
    U_t: torch.Tensor,
) -> torch.Tensor:
    """Build the A matrix for the CD linear problem Ax=b (up to 2-body terms).

    Uses index_put (non-in-place) so gradients flow through Omega_t, Mu_t,
    Nu_t.
    U_t (ZZ interaction) is detached — it is fixed by hardware geometry, not
    optimized.
    """
    n_single = 3 * n_atoms
    n_sym = 3 * len(list(combinations(range(n_atoms), 2)))
    n_asym = 3 * len(list(permutations(range(n_atoms), 2)))
    n_total = n_single + n_sym + n_asym
    asym_start = n_single + n_sym  # sing_plus_sym

    rows: list[int] = []
    cols: list[int] = []
    vals: list[torch.Tensor] = []

    # single-body block
    for i in range(n_atoms):
        rows += [1 + 3 * i, 0 + 3 * i, 0 + 3 * i]
        cols += [2 + 3 * i, 2 + 3 * i, 1 + 3 * i]
        vals += [-Omega_t[i], Mu_t[i], -Nu_t[i]]

    # ZZ interaction → asymmetric 2-body columns
    asym_col = 2
    for i, j in combinations(range(n_atoms), 2):
        u = U_t[i, j].detach()  # fixed by hardware, not a free parameter
        rows += [i * 3, i * 3 + 1, j * 3, j * 3 + 1]
        cols += [
            asym_start + asym_col,
            asym_start + asym_col - 1,
            asym_start + asym_col + 3,
            asym_start + asym_col + 2,
        ]
        vals += [-u, u, -u, u]
        asym_col += 6

    # symmetric 2-body × single-body coupling
    asym_col, block_offset = 0, 0
    for i, j in combinations(range(n_atoms), 2):
        r = n_single + block_offset
        rows += [r, r, r, r, r + 1, r + 1, r + 1, r + 1, r + 2, r + 2, r + 2, r + 2]
        cols += [
            asym_start + asym_col,
            asym_start + asym_col + 3,
            asym_start + asym_col + 1,
            asym_start + asym_col + 4,
            asym_start + asym_col,
            asym_start + asym_col + 3,
            asym_start + asym_col + 2,
            asym_start + asym_col + 5,
            asym_start + asym_col + 1,
            asym_start + asym_col + 4,
            asym_start + asym_col + 2,
            asym_start + asym_col + 5,
        ]
        vals += [
            -Nu_t[j],
            -Nu_t[i],
            Mu_t[j],
            Mu_t[i],
            Nu_t[i],
            Nu_t[j],
            -Omega_t[j],
            -Omega_t[i],
            -Mu_t[i],
            -Mu_t[j],
            Omega_t[i],
            Omega_t[j],
        ]
        asym_col += 6
        block_offset += 3

    # asymmetric 2-body self-coupling
    asym_col, block_offset = 1, 0
    for i, j in combinations(range(n_atoms), 2):
        r = asym_start + block_offset
        rows += [r, r + 3, r + 1, r + 4, r + 2, r]
        cols += [
            asym_start + asym_col,
            asym_start + asym_col + 3,
            asym_start + asym_col + 1,
            asym_start + asym_col + 4,
            asym_start + asym_col + 2,
            asym_start + asym_col + 4,
        ]
        vals += [-Omega_t[j], -Omega_t[i], -Nu_t[i], -Nu_t[j], -Mu_t[j], Mu_t[i]]
        asym_col += 6
        block_offset += 6

    # assemble via index_put (differentiable w.r.t. vals)
    vals_t = torch.stack(vals)
    A_upper = torch.zeros(n_total, n_total, dtype=dtype).index_put(
        (torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)),
        vals_t,
    )
    return A_upper - A_upper.T


def b_direct_vec(
    n_atoms: int, dOmega_t: torch.Tensor, dMu_t: torch.Tensor, dNu_t: torch.Tensor
) -> torch.Tensor:
    n_sym = 3 * len(list(combinations(range(n_atoms), 2)))
    n_asym = 3 * len(list(permutations(range(n_atoms), 2)))

    singles = torch.stack(
        [v for i in range(n_atoms) for v in (-dOmega_t[i], -dMu_t[i], -dNu_t[i])]
    )
    rest = torch.zeros(n_sym + n_asym, dtype=dtype)
    return torch.cat([singles, rest])


def solve_cd_torch(M: torch.Tensor, b: torch.Tensor, reg: float = 1e-4) -> torch.Tensor:
    """
    Minimum-norm least-squares solution via pseudo-inverse (differentiable).
    """
    M = M + reg * torch.eye(M.shape[0], dtype=M.dtype, device=M.device)

    return torch.linalg.lstsq(M, b).solution
