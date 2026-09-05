"""Numerically robust whitening, installed over lems's implementation.

Transferring the pipeline to Qwen2-VL surfaced a failure that never occurred on
LLaVA: ``torch.linalg.eigvalsh`` on the GPU raises

    linalg.eigh: The algorithm failed to converge because the input matrix is
    ill-conditioned or has too many repeated eigenvalues

inside lems's Cholesky whitening. That call is on the recovery path taken when
the covariance is not positive definite, so a matrix that merely needed a shift
instead terminates the run. lems's own handler then calls ``sys.exit()`` after
three attempts, which kills the job without a traceback.

Two hazards are addressed here, both of them documented properties of cuSOLVER
on this hardware rather than bugs in lems:

* an eigendecomposition that will not converge on the GPU generally succeeds on
  the CPU in float64, which uses a different LAPACK path;
* a Cholesky factor can be returned alongside a success code while containing
  non-finite entries, so the factor has to be inspected rather than trusted.

``install_robust_whitening()`` replaces the module-level function that lems's
own ``whitening`` dispatcher calls, so every factorization method picks it up.
The mathematics is unchanged: the same shift-and-retry scheme, computed
somewhere it converges.
"""

from __future__ import annotations

import torch

_installed = False


def _safe_eigvalsh(matrix: torch.Tensor, name: str) -> torch.Tensor:
    """Smallest-first eigenvalues, retrying on CPU in float64 if cuSOLVER fails."""
    if not torch.isfinite(matrix).all():
        # Non-finite entries make the decomposition fail on every backend, and
        # the resulting error names linear algebra rather than the real cause,
        # which is upstream: a covariance accumulated in a precision the model
        # overflows in. Say so here rather than let it surface as convergence.
        bad = int((~torch.isfinite(matrix)).sum())
        raise ValueError(
            f"covariance for {name!r} has {bad} non-finite entries; the "
            "calibration statistics overflowed. Check the model is loaded in "
            "the precision its checkpoint specifies (bfloat16 checkpoints "
            "overflow in float16)."
        )
    try:
        values = torch.linalg.eigvalsh(matrix)
        if torch.isfinite(values).all():
            return values
        reason = "non-finite eigenvalues"
    except torch.linalg.LinAlgError as exc:
        reason = str(exc).split("\n", 1)[0]
    print(f"  [numerics] GPU eigvalsh failed for {name!r} ({reason}); "
          f"retrying on CPU in float64")
    return torch.linalg.eigvalsh(matrix.detach().cpu().double()).to(matrix.device)


def _robust_whitening_cholesky(dev, raw_scale, name, alpha=0.0, increment=1e-6,
                               *, double_precision=False):
    """Drop-in replacement for lems's ``_whitening_cholesky``."""
    if double_precision:
        matrix = raw_scale[name].double().to(dev)
    else:
        matrix = raw_scale[name].clone().float().to(dev)

    if alpha > 0.0:
        shrinkage = torch.mean(matrix.diag()) * alpha
        matrix.mul_(1.0 - alpha)
        matrix.diagonal().add_(shrinkage)

    factor = None
    for attempt in range(4):
        try:
            candidate = torch.linalg.cholesky(matrix)
            # A success code does not guarantee a usable factor.
            if torch.isfinite(candidate).all():
                factor = candidate
                break
            raise torch.linalg.LinAlgError("non-finite Cholesky factor")
        except torch.linalg.LinAlgError:
            if attempt == 3:
                break
            eigenvalues = _safe_eigvalsh(matrix, name)
            matrix.diagonal().add_(-eigenvalues[0].to(matrix.dtype) + increment)
            del eigenvalues

    if factor is None:
        # Last resort: whiten from the eigendecomposition instead. Slower, but
        # it neither requires positive definiteness nor aborts the run, which
        # is what lems does here.
        print(f"  [numerics] Cholesky exhausted for {name!r}; "
              f"falling back to eigendecomposition whitening")
        symmetric = matrix.detach().cpu().double()
        symmetric = 0.5 * (symmetric + symmetric.T)
        values, vectors = torch.linalg.eigh(symmetric)
        values = values.clamp_min(increment)
        root = vectors @ torch.diag(values.sqrt()) @ vectors.T
        inverse_root = vectors @ torch.diag(values.rsqrt()) @ vectors.T
        return (root.to(dev).float(), inverse_root.to(dev).float())

    identity = torch.eye(factor.shape[0], device=dev, dtype=factor.dtype)
    factor_inverse = torch.linalg.solve_triangular(factor, identity, upper=False)
    return factor.float(), factor_inverse.float()


def install_robust_whitening() -> None:
    """Point lems's whitening dispatcher at the robust implementation."""
    global _installed
    if _installed:
        return
    import compression.factorization._interface as interface

    interface._whitening_cholesky = _robust_whitening_cholesky
    _installed = True
    print("[numerics] robust whitening installed over lems's implementation")
