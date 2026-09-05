"""Solve the LEMS rank-allocation ILP with HiGHS.

lems ships two solvers and neither is a good fit here:

* **Gurobi** (its default) needs a commercial licence. The size-limited free
  licence rejects this model outright -- measured on LLaVA-1.5-7B, the
  untrimmed problem is 540,920 binary variables across 364 layers, and even
  lems's own default granularity (rank multiples of 8) leaves ~67,700.
* **CBC**, its free fallback, is carried by lems with the explicit caveat that
  it may not match Gurobi's solutions, and its README recommends coarsening
  the rank grid to multiples of 64 to keep CBC tractable -- which throws away
  most of the allocation's resolution.

HiGHS is open source (MIT), needs no licence, and is a far stronger MIP solver
than CBC, so it recovers the fine rank grid without the licence problem.

The model itself is a multiple-choice knapsack, identical in form to lems's:
minimise total weighted error, subject to a global parameter budget and
exactly one rank chosen per layer. Only the monotonicity extras lems leaves
disabled are omitted.
"""

from __future__ import annotations

import pulp


def ilp_search_highs(data, compression_list, layer_name_list,
                     compression_param_target, ilp_settings=None,
                     layers_per_block=None, rank_list=None,
                     shared_rank_groups=None, time_limit_seconds: int = 300,
                     msg: bool = False):
    """Choose one rank per layer to minimise total error under a budget.

    Parameters mirror ``lems.compression.search.lems.ilp_search_cbc`` so this
    is a drop-in replacement.

    ``data[i]`` is a list of ``(retained_parameters, error)`` pairs for layer
    ``i``; ``compression_list[i]`` holds the corresponding retained-parameter
    ratios. Returns ``{layer_name: chosen_ratio}``.
    """
    model = pulp.LpProblem("Minimize_Compression_Error", pulp.LpMinimize)

    variables = [
        [pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(len(choices))]
        for i, choices in enumerate(data)
    ]

    model += pulp.lpSum(
        data[i][j][1] * variables[i][j]
        for i in range(len(data)) for j in range(len(data[i]))
    ), "Total_Error"

    model += pulp.lpSum(
        data[i][j][0] * variables[i][j]
        for i in range(len(data)) for j in range(len(data[i]))
    ) <= compression_param_target, "Parameter_Budget"

    for i in range(len(data)):
        model += pulp.lpSum(variables[i]) == 1, f"Select_One_From_Var_{i}"

    solver = pulp.HiGHS(timeLimit=time_limit_seconds, msg=msg)
    model.solve(solver)

    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        # A truncated or infeasible solve would silently produce a bad rank
        # vector that still "works", so surface it rather than continue.
        raise RuntimeError(
            f"HiGHS returned status {status!r} for the rank-allocation ILP "
            f"({len(data)} layers, budget {compression_param_target:.0f}). "
            "Raise time_limit_seconds or coarsen the rank grid."
        )

    compression_dict = {}
    total_params = 0.0
    total_error = 0.0
    for i in range(len(data)):
        for j in range(len(data[i])):
            value = variables[i][j].varValue
            if value is not None and value > 0.99:
                compression_dict[layer_name_list[i]] = compression_list[i][j]
                total_params += data[i][j][0]
                total_error += data[i][j][1]
                break

    print(f"HiGHS: {status}, error={total_error:.4f}, "
          f"params={total_params:.0f}/{compression_param_target:.0f} "
          f"({len(compression_dict)}/{len(data)} layers assigned)")
    return compression_dict
