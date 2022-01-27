#!/usr/bin/env python3

import numpy as np

def poisson_binomial_pmf(probabilities: np.ndarray) -> np.ndarray:
    '''Poisson-Binomial probability mass function.

    Args:
        probabilities (np.ndarray): Probability of success of each trial.

    Returns:
        np.ndarray: Sequence where $i$th element is probability of observing exactly $i$
        successful trials.
    '''
    alpha = np.prod(probabilities)
    s = -(1 - probabilities) / probabilities
    coeffs = np.poly(s) * alpha
    coeffs = np.flip(coeffs)
    return coeffs


def mttdl_formula(n: int, k: int, failure_rates: np.ndarray, repair_rate: float) -> float:
    '''Calculate MTTDL via an approximate formula.

    Args:
        n (int): Length of MDS code.
        k (int): Dimension of MDS code.
        failure_rates (np.ndarray): Array of length `n + 1`. First `n` elements correspond to
            failure rates of disks.
        repair_rate (float): Repair rate of disks.

    Returns:
        float: approximate MTTDL.
    '''
    probabilities = repair_rate / (repair_rate + failure_rates)
    pmf = poisson_binomial_pmf(probabilities)
    data_loss_rate = repair_rate * (n - k + 1) * pmf[k - 1]
    return 1 / data_loss_rate


if __name__ == '__main__':
    n = 14
    k = 10
    failure_rates = np.array([2.07e-06] * 7 + [4.57e-06] * 7 + [4.])
    repair_rate = 4.
    mttdl = mttdl_formula(n, k, failure_rates, repair_rate)
    print(
        f'MTTDL(n={n}, k={k}, failure rates={failure_rates}, '
        f'repair rate={repair_rate}): {mttdl}')
