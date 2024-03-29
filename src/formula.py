#!/usr/bin/env python3

from typing import Tuple
import numpy as np

from scipy.special import binom
from scipy.optimize import root_scalar


def poisson_binomial_pmf(probabilities: np.ndarray) -> np.ndarray:
    """Poisson-Binomial probability mass function.

    Args:
        probabilities (np.ndarray): Probability of success of each trial.

    Returns:
        np.ndarray: Sequence where $i$th element is probability of observing exactly $i$
        successful trials.
    """
    alpha = np.prod(probabilities)
    s = -(1 - probabilities) / probabilities
    coeffs = np.poly(s) * alpha
    coeffs = np.flip(coeffs)
    return coeffs


def homogeneous_mttdl_formula(
    n: int, k: int, failure_rate: float, repair_rate: float
) -> float:
    """Calculate homogeneous MTTDL via an approximate formula.

    Args:
        n (int): Length of MDS code.
        k (int): Dimension of MDS code.
        failure_rate (float): Failure rate of disks.
        repair_rate (float): Repair rate of disks.

    Returns:
        float: approximate MTTDL.
    """
    nck = binom(n, k)
    ratio = (failure_rate / repair_rate) ** (n - k)
    data_loss_rate = nck * ratio * failure_rate * k
    return 1 / data_loss_rate


def mttdl_formula(
    n: int, k: int, failure_rates: np.ndarray, repair_rate: float
) -> float:
    """Calculate MTTDL via an approximate formula.

    Args:
        n (int): Length of MDS code.
        k (int): Dimension of MDS code.
        failure_rates (np.ndarray): Array of length `n + 1`. First `n` elements correspond to
            failure rates of disks.
        repair_rate (float): Repair rate of disks.

    Returns:
        float: approximate MTTDL.
    """
    probabilities = repair_rate / (repair_rate + failure_rates)
    pmf = poisson_binomial_pmf(probabilities)
    data_loss_rate = repair_rate * (n - k + 1) * pmf[k - 1]
    return 1 / data_loss_rate


def critical_afr(
    n: int,
    k: int,
    mttdl_threshold: float,
    repair_rate: float,
    bracket: Tuple[float, float],
) -> float:
    """Find the highest failure rate that can be tolerated by a given scheme.

    Args:
        n (int): Length of MDS code.
        k (int): Dimension of MDS code.
        mttdl_threshold (float): minimum MTTDL threshold.
        repair_rate (float): Repair rate of disks.
        bracket (Tuple[float, float]): Failure rate bracket (must contain solution).

    Returns:
        float: Failure rate.
    """
    root_fun = (
        lambda afr: homogeneous_mttdl_formula(n, k, afr, repair_rate) - mttdl_threshold
    )
    sol = root_scalar(root_fun, bracket=bracket)
    return sol.root


def _test_homogeneous_mttdl_formula():
    """Test that homogeneous and heterogenous formula give similar results."""
    n = 14
    k = 10
    failure_rate = 4.57e-06
    failure_rates = np.array([failure_rate] * n)
    repair_rate = 4.0
    homogeneous_mttdl = homogeneous_mttdl_formula(n, k, failure_rate, repair_rate)
    mttdl = mttdl_formula(n, k, failure_rates, repair_rate)
    print(
        f"homogeneous_MTTDL(n={n}, k={k}, failure rate={failure_rate}, "
        f"repair rate={repair_rate}): {homogeneous_mttdl}"
    )
    print(
        f"MTTDL(n={n}, k={k}, failure rates={failure_rates}, "
        f"repair rate={repair_rate}): {mttdl}"
    )
    assert abs(np.log(homogeneous_mttdl) - np.log(mttdl)) < 1


def _test_mttdl_formula():
    """Test that MTTDL formula gives sane results."""
    n = 14
    k = 10
    failure_rates = np.array([2.07e-06] * 7 + [4.57e-06] * 7)
    repair_rate = 4.0
    mttdl = mttdl_formula(n, k, failure_rates, repair_rate)
    print(
        f"MTTDL(n={n}, k={k}, failure rates={failure_rates}, "
        f"repair rate={repair_rate}): {mttdl}"
    )
    assert abs(np.log(mttdl) - np.log(7e25)) < 1


def _test_critical_afr():
    """Test that critical AFR is found correctly."""
    n = 14
    k = 10
    repair_rate = 4.0
    min_afr = 2 / (365 * 24 * 100)
    max_afr = 15 / (365 * 24 * 100)
    mttdl_threshold = homogeneous_mttdl_formula(n, k, max_afr, repair_rate)
    bracket = (min_afr, max_afr)
    afr = critical_afr(n, k, mttdl_threshold, repair_rate, bracket)
    print(f"afr={max_afr}")
    print(
        f"critical_afr(n={n}, k={k}, mttdl_threshold={mttdl_threshold}, "
        f"repair rate={repair_rate}, bracket={bracket}): {afr}"
    )
    assert np.isclose(afr, max_afr, rtol=1e-3)


if __name__ == "__main__":
    # Run tests.
    _test_mttdl_formula()
    _test_homogeneous_mttdl_formula()
    _test_critical_afr()
