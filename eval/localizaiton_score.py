from collections import defaultdict

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from diagnoser.metric_node import MetricNodes
from eval.groundtruth import check_cause_metrics
from meltria.priorknowledge.priorknowledge import PriorKnowledge


def get_ranks_by_case(sorted_results_df: DataFrameGroupBy, pk: PriorKnowledge, granularity: str = "metric"):
    ranks_by_case: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    n_cases: int = 0
    for (dataset_id, target_app, chaos_type, chaos_comp, chaos_case_num), row in sorted_results_df:
        if chaos_comp in pk.get_skip_containers():
            continue
        metrics = [str(m) for m in row["metric_name"].values.tolist()]
        ranks: list[int]
        match granularity:
            case "metric":
                ok, cause_metrics = check_cause_metrics(
                    pk,
                    metrics,
                    chaos_type=chaos_type,
                    chaos_comp=chaos_comp,
                    optional_cause=True,
                )
                if not ok or len(cause_metrics) == 0:
                    print(f"no cause metrics: {dataset_id}, {target_app}, {chaos_type}, {chaos_comp}")
                    continue
                metrics = [m for m in metrics if not m.startswith("s-")]  # Exclude service metrics
                ranked_metrics = MetricNodes.from_metric_names(metrics)
                ranks = sorted([list(ranked_metrics).index(cm) + 1 for cm in cause_metrics])
            case "container":
                metrics = [m for m in metrics if not m.startswith("s-")]  # Exclude service metrics
                ranked_ctnrs = list(set([pk.get_container_by_metric(metric) for metric in metrics]))
                ranks = sorted([i + 1 for i, ctnr in enumerate(ranked_ctnrs) if ctnr == chaos_comp])
            case "service":
                chaos_service: str = pk.get_service_by_container(chaos_comp)
                ranked_service = list(set([pk.get_service_by_metric(metric) for metric in metrics]))
                ranked_service = [s for s in ranked_service if s is not None and not s.startswith("gke-")]
                ranks = sorted([i + 1 for i, service in enumerate(ranked_service) if service == chaos_service])
            case _:
                assert False, f"Unknown detect_unit: {granularity}"
        ranks_by_case[(chaos_type, chaos_comp, chaos_case_num)] = ranks
        n_cases += 1
    return ranks_by_case, n_cases


def calc_ac_k(k: int, ranks_by_case: dict[tuple[str, str, int], list[int]], n_faults: int) -> float:
    if n_faults == 0:
        return 0.0
    sum_ac = 0.0
    for _, ranks in ranks_by_case.items():
        if min_param := min(k, len(ranks)) > 0:
            sum_ac += sum([1 if ranks[i - 1] <= k else 0 for i in range(1, min_param + 1)]) / min_param
    return sum_ac / n_faults


def evaluate_ac_of_rc(
    sorted_results_df: DataFrameGroupBy,
    pk: PriorKnowledge,
    k: int = 10,
    granuallity: str = "metric",
) -> pd.DataFrame:
    top_k_set = range(1, k + 1)
    ranks_by_case, n_cases = get_ranks_by_case(sorted_results_df, pk, granularity=granuallity)
    ac_k = {k: calc_ac_k(k, ranks_by_case, n_cases) for k in top_k_set}
    avg_k = {k: sum([ac_k[j] for j in range(1, k + 1)]) / k for k in top_k_set}
    return pd.concat(
        [
            pd.DataFrame({k: n_cases for k in top_k_set}, index=[f"#cases ({granuallity})"]).T,
            pd.DataFrame(ac_k, index=[f"AC@K ({granuallity})"]).T,
            pd.DataFrame(avg_k, index=[f"AVG@K ({granuallity})"]).T,
        ],
        axis=1,
    )
