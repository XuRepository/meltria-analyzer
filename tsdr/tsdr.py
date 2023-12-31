import time
from collections.abc import Callable
from concurrent import futures
from functools import partial
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.signal
import scipy.stats
from joblib import Parallel, delayed
from pandarallel import pandarallel
from scipy.spatial.distance import hamming

import tsdr.multireducer as multireducer
import tsdr.unireducer as unireducer
from eval.validation import detect_anomalies_with_spot
from meltria.loader import count_metrics
from meltria.priorknowledge.priorknowledge import PriorKnowledge
from tsdr import smooth
from tsdr.birch import detect_anomalies_with_birch
from tsdr.clustering.pearsonr import pearsonr_as_dist
from tsdr.clustering.sbd import sbd
from tsdr.outlierdetection.n_sigma_rule import detect_with_n_sigma_rule
from tsdr.unireducer import UnivariateSeriesReductionResult

ELAPSED_TIME_NUM_DECIMAL_PLACES: Final[int] = 4

pandarallel.initialize(progress_bar=False, verbose=0)


class Tsdr:
    def __init__(
        self,
        univariate_series_func_or_name: Callable[
            [np.ndarray, Any], UnivariateSeriesReductionResult
        ]
        | str,
        enable_unireducer: bool = True,
        enable_multireducer: bool = True,
        **kwargs: Any,
    ) -> None:
        self.params = kwargs
        self.enable_unireducer = enable_unireducer
        self.enable_multireducer = enable_multireducer
        if callable(univariate_series_func_or_name):
            setattr(self, "univariate_series_func", univariate_series_func_or_name)
        elif type(univariate_series_func_or_name) == str:
            self.univariate_model_name = univariate_series_func_or_name
            match univariate_series_func_or_name:
                case "birch_model":
                    pass
                case _:
                    func: Callable = unireducer.map_model_name_to_func(
                        univariate_series_func_or_name
                    )
                    setattr(self, "univariate_series_func", func)
        else:
            raise TypeError(
                f"Invalid type of step1 mode: {type(univariate_series_func_or_name)}"
            )

    def univariate_series_func(
        self, series: np.ndarray, **kwargs: Any
    ) -> UnivariateSeriesReductionResult:
        return unireducer.ar_based_ad_model(series, **kwargs)

    def reduce_by_failure_detection_time(
        self,
        series: pd.DataFrame,
        results: dict[str, UnivariateSeriesReductionResult],
        sli: np.ndarray,
        sigma_threshold: int = 3,
    ) -> pd.DataFrame:
        """reduce series by failure detection time"""
        outliers = detect_with_n_sigma_rule(
            x=sli,
            test_start_time=self.params["time_fault_inject_time_index"],
            sigma_threshold=sigma_threshold,
        )
        failure_detection_time: int = 0 if outliers.size == 0 else outliers[0]

        kept_series_labels: list[str] = []
        for resuts_col, res in results.items():
            if not res.has_kept:
                continue
            change_start_time = res.change_start_point[0]
            if (
                failure_detection_time == 0
                or failure_detection_time >= change_start_time
            ):
                kept_series_labels.append(resuts_col)
        return series[kept_series_labels]

    def get_most_anomalous_sli(self, series: pd.DataFrame, slis: list[str]) -> str:
        assert len(slis) > 0, "slis is empty"
        sli_df: pd.DataFrame = series.loc[
            :, [col for col in slis if col in series.columns]
        ]  # retrieve only existing slis
        idx = self.params["sli_anomaly_start_time_index"]
        return sli_df.apply(
            lambda x: detect_anomalies_with_spot(x.to_numpy(), idx)[1]
        ).idxmax()

    # def corrs_with_sli(
    #     self, series: pd.DataFrame, slis: list[str], left_shit: bool = False, l_p: int = 0
    # ) -> pd.Series:
    #     assert not (left_shit and l_p == 0)
    #     sli_name = self.get_most_anomalous_sli(series, slis)
    #     sli = series[sli_name].to_numpy()
    #     if left_shit:
    #         return series.apply(lambda x: pearsonr_left_shift(x.to_numpy(), sli, apply_abs=True, l_p=l_p), axis=0)
    #     return series.apply(lambda x: pearsonr(x.to_numpy(), sli, apply_abs=True), axis=0)

    def run(
        self,
        X: pd.DataFrame,
        pk: PriorKnowledge,
        max_workers: int,
    ) -> tuple[
        list[tuple[pd.DataFrame, pd.DataFrame, float]],
        pd.DataFrame,
        dict[str, np.ndarray],
    ]:
        assert X.columns.size == np.unique(X.columns).size

        stat: list[tuple[pd.DataFrame, pd.DataFrame, float]] = []
        stat.append((X, count_metrics(X), 0.0))

        # step0
        start: float = time.time()

        series: pd.DataFrame = filter_out_no_change_metrics(
            X, parallel=(max_workers != 1)
        )

        elapsed_time: float = round(
            time.time() - start, ELAPSED_TIME_NUM_DECIMAL_PLACES
        )
        stat.append((series, count_metrics(series), elapsed_time))

        # step1
        anomaly_points: dict[str, np.ndarray] = {}
        step1_results: dict = {}
        if self.enable_unireducer:
            start = time.time()

            _series: pd.DataFrame = series
            if self.params.get("step1_enable_smoother", False):
                _series = series.apply(
                    lambda x: smooth.moving_average(
                        x, window_size=self.params["step1_smoother_window_size"]
                    )
                )

            match self.univariate_model_name:
                case "pearsonr_sli":
                    reduced_series1 = self.reduce_univariate_series_to_sli(
                        _series, pk, max_workers
                    )
                case "birch_model":
                    birch_res = detect_anomalies_with_birch(_series, **self.params)
                    reduced_series1 = _series[
                        [
                            metric
                            for metric, is_normal in birch_res.items()
                            if not is_normal
                        ]
                    ]
                case _:
                    (
                        reduced_series1,
                        step1_results,
                        anomaly_points,
                    ) = self.reduce_univariate_series(_series, pk, max_workers)

            elapsed_time = round(time.time() - start, ELAPSED_TIME_NUM_DECIMAL_PLACES)
            stat.append((reduced_series1, count_metrics(reduced_series1), elapsed_time))

            if self.params.get("step1_residual_integral_change_start_point", False):
                # step1.5
                start = time.time()

                sli_name: str = pk.get_root_metrics()[
                    0
                ]  # TODO: choose SLI metrics formally
                reduced_series1 = self.reduce_by_failure_detection_time(
                    reduced_series1,
                    step1_results,
                    series[sli_name].to_numpy(),
                    sigma_threshold=self.params[
                        "step1_residual_integral_change_start_point_n_sigma"
                    ],
                )

                elapsed_time = round(
                    time.time() - start, ELAPSED_TIME_NUM_DECIMAL_PLACES
                )
                stat.append(
                    (reduced_series1, count_metrics(reduced_series1), elapsed_time)
                )
        else:
            reduced_series1 = series

        # step2
        clusters_stat: pd.DataFrame = pd.DataFrame()
        if self.enable_multireducer:
            start = time.time()

            df_before_clustering = self.preprocess_for_multireducer(
                reduced_series1, step1_results, max_workers
            )
            reduced_series2, clusters_stat = self.reduce_multivariate_series(
                df_before_clustering,
                pk=pk,
                step1_results=step1_results,
                n_workers=max_workers,
            )

            elapsed_time = round(time.time() - start, ELAPSED_TIME_NUM_DECIMAL_PLACES)
            stat.append((reduced_series2, count_metrics(reduced_series2), elapsed_time))

        return stat, clusters_stat, anomaly_points

    def reduce_univariate_series(
        self,
        useries: pd.DataFrame,
        pk: PriorKnowledge,
        n_workers: int,
    ) -> tuple[
        pd.DataFrame, dict[str, UnivariateSeriesReductionResult], dict[str, np.ndarray]
    ]:
        results: dict[str, UnivariateSeriesReductionResult] = {}
        # TODO: replace futures with joblib
        with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_col = {}
            for col in useries.columns:
                series: np.ndarray = useries[col].to_numpy()
                future = executor.submit(
                    self.univariate_series_func, series, **self.params
                )
                future_to_col[future] = col
            reduced_cols: list[str] = []
            for future in futures.as_completed(future_to_col):
                col = future_to_col[future]
                result: UnivariateSeriesReductionResult = future.result()
                results[col] = result
                if result.has_kept:
                    reduced_cols.append(col)
        anomaly_points = {col: res.outliers for col, res in results.items()}
        return useries[reduced_cols], results, anomaly_points

    def reduce_univariate_series_to_sli(
        self, useries: pd.DataFrame, pk: PriorKnowledge, n_workers: int
    ) -> pd.DataFrame:
        sli = self.get_most_anomalous_sli(useries, list(pk.get_root_metrics()))
        sli_data = useries[sli].to_numpy()
        results = Parallel(n_jobs=n_workers)(
            delayed(self.univariate_series_func)(
                useries[col].to_numpy(), sli_data, **self.params
            )
            for col in useries.columns
        )
        assert results is not None
        reduced_cols: list[str] = [
            col for col, result in zip(useries.columns, results) if result.has_kept
        ]
        return useries[reduced_cols]

    def preprocess_for_multireducer(
        self,
        series: pd.DataFrame,
        step1_results: dict[str, UnivariateSeriesReductionResult],
        max_workers: int = 1,
    ) -> pd.DataFrame:
        match series_type := self.params["step2_clustering_series_type"]:
            case "raw":
                _series: pd.DataFrame = series
                if self.params.get("step2_enable_smoother", False):
                    _series = series.apply(
                        lambda x: smooth.moving_average(
                            x, window_size=self.params["step2_smoother_window_size"]
                        )
                    )
                    # filter metrics including nan values after zscore
                    _series = filter_out_no_change_metrics(
                        _series, parallel=(max_workers != 1)
                    )
                preprocessed_df = _series.apply(scipy.stats.zscore)
            case "anomaly_score" | "binary_anomaly_score":
                tmp_dict_to_df: dict[str, np.ndarray] = {}
                for name, res in step1_results.items():
                    if res.has_kept:
                        if series_type == "anomaly_score":
                            tmp_dict_to_df[name] = scipy.stats.zscore(
                                res.anomaly_scores
                            )
                        elif series_type == "binary_anomaly_score":
                            tmp_dict_to_df[name] = res.binary_scores()
                preprocessed_df = pd.DataFrame(tmp_dict_to_df)
            case _:
                raise ValueError(
                    f"step2_clustered_series_type is invalid {series_type}"
                )

        assert (
            preprocessed_df.isnull().values.sum() == 0
        ), "df_before_clustering has nan values"
        return preprocessed_df

    def reduce_multivariate_series(
        self,
        series: pd.DataFrame,
        pk: PriorKnowledge,
        step1_results: dict[str, UnivariateSeriesReductionResult],
        granularity: str = "container",
        n_workers: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        n_workers = self.params.get("step2_clustering_n_workers", n_workers)
        granularity = self.params.get("step2_clustering_granularity", granularity)

        sli_data: np.ndarray | None = None
        if self.params["step2_clustering_choice_method"] in [
            "cluster_of_max_corr_to_sli",
            "nearest_sli_changepoint",
        ]:
            sli = self.get_most_anomalous_sli(series, list(pk.get_root_metrics()))
            sli_data = series[sli].to_numpy()

        clusterinf_fn: Callable = self._prepare_clustering_fn(
            series, step1_results, sli_data, **self.params
        )
        comp_to_metrics_df: dict[str, pd.DataFrame] = {}
        clusters_stats: list[dict[str, Any]] = []

        match granularity:
            case "service":
                for service, containers in pk.get_containers_of_service().items():
                    metrics_dfs: list[pd.DataFrame] = []

                    service_metrics_df = series.loc[
                        :, series.columns.str.startswith(f"s-{service}_")
                    ]
                    if len(service_metrics_df.columns) > 0:
                        metrics_dfs.append(service_metrics_df)

                    for container in containers:
                        container_metrics_df = series.loc[
                            :,
                            series.columns.str.startswith(
                                (f"c-{container}_", f"m-{container}_")
                            ),
                        ]
                        if len(container_metrics_df.columns) > 0:
                            metrics_dfs.append(container_metrics_df)

                    if len(metrics_dfs) > 0:
                        comp_to_metrics_df[service] = pd.concat(metrics_dfs, axis=1)

            case "container":
                # Clustering metrics by service including services, containers and middlewares metrics
                for service, containers in pk.get_containers_of_service().items():
                    # 1. service-level clustering
                    # TODO: retrieve service metrics efficently
                    service_metrics_df = series.loc[
                        :, series.columns.str.startswith(f"s-{service}_")
                    ]
                    if len(service_metrics_df.columns) > 1:
                        comp_to_metrics_df[f"s-{service}"] = service_metrics_df

                    # 2. container-level clustering
                    for container in containers:
                        # perform clustering in each type of metric
                        # TODO: retrieve container and middleware metrics efficently
                        container_metrics_df = series.loc[
                            :,
                            series.columns.str.startswith(
                                (f"c-{container}_", f"m-{container}_")
                            ),
                        ]
                        if len(container_metrics_df.columns) <= 1:
                            continue
                        comp_to_metrics_df[f"c-{container}"] = container_metrics_df
            case _:
                assert False, f"Invalid granularity: {granularity}"

        # 3. node-level clustering
        for node in pk.get_nodes():
            node_metrics_df = series.loc[:, series.columns.str.startswith(f"n-{node}_")]
            if len(node_metrics_df.columns) <= 1:
                continue
            comp_to_metrics_df[f"n-{node}"] = node_metrics_df

        clustering_results = []
        if n_workers == 1:
            for comp, metrics_df in comp_to_metrics_df.items():
                c_info, remove_list = clusterinf_fn(metrics_df)
                clustering_results.append((comp, c_info, remove_list))
        else:
            with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_comp: dict[futures.Future, str] = {
                    executor.submit(clusterinf_fn, metrics_df): comp
                    for comp, metrics_df in comp_to_metrics_df.items()
                }
                for future in futures.as_completed(future_to_comp):
                    comp = future_to_comp[future]
                    c_info, remove_list = future.result()
                    clustering_results.append((comp, c_info, remove_list))

        for comp, c_info, remove_list in clustering_results:
            for rep_metric, sub_metrics in c_info.items():
                clusters_stats.append(
                    {
                        "component": comp,
                        "rep_metric": rep_metric,
                        "sub_metrics": sub_metrics,
                    }
                )
            series.drop(remove_list, axis=1, inplace=True)
        return series, pd.DataFrame(clusters_stats)

    @staticmethod
    def _prepare_clustering_fn(
        series: pd.DataFrame,
        step1_results: dict[str, UnivariateSeriesReductionResult],
        sli_data: np.ndarray | None,
        **kwargs: Any
    ) -> Callable:
        method_name = kwargs["step2_clustering_method_name"]
        choice_method = kwargs["step2_clustering_choice_method"]

        match method_name:
            case "hierarchy":
                dist_type = kwargs["step2_hierarchy_dist_type"]
                dist_threshold = kwargs["step2_hierarchy_dist_threshold"]
                linkage_method = kwargs["step2_hierarchy_linkage_method"]

                dist_func: str | Callable
                match dist_type:
                    case "sbd":
                        dist_func = sbd
                    case "pearsonr":
                        dist_func = pearsonr_as_dist
                    case "hamming":
                        if dist_threshold >= 1.0:
                            # make the distance threshold intuitive
                            dist_threshold /= series.shape[0]
                        dist_func = hamming
                    case _:
                        dist_func = dist_type

                return partial(
                    multireducer.hierarchical_clustering,
                    dist_func=dist_func,
                    dist_threshold=dist_threshold,
                    choice_method=choice_method,
                    linkage_method=linkage_method,
                    sli_data=sli_data,
                    top_k=kwargs.get("step2_clustering_choice_top_k", 1),
                )
            case "dbscan":
                dist_type = kwargs["step2_dbscan_dist_type"]
                match dist_type:
                    case "sbd":
                        dist_func = sbd
                    case "hamming":
                        dist_func = hamming
                    case "pearsonr":
                        dist_func = pearsonr_as_dist
                    case _:
                        dist_func = dist_type

                return partial(
                    multireducer.dbscan_clustering,
                    dist_func=dist_func,
                    min_pts=kwargs["step2_dbscan_min_pts"],
                    algorithm=kwargs["step2_dbscan_algorithm"],
                    eps=kwargs.get("step2_dbscan_eps", 0.0),
                    choice_method=choice_method,
                    sli_data=sli_data,
                    top_k=kwargs.get("step2_clustering_choice_top_k", 1),
                )
            case "changepoint":
                return partial(
                    multireducer.change_point_clustering,
                    cost_model=kwargs["step2_changepoint_cost_model"],
                    n_bkps=kwargs["step2_changepoint_n_bkps"],
                    proba_threshold=kwargs["step2_changepoint_proba_threshold"],
                    choice_method=choice_method,
                    cluster_selection_method=kwargs.get(
                        "step2_changepoint_cluster_selection_method", "leaf"
                    ),
                    cluster_selection_epsilon=kwargs.get(
                        "step2_changepoint_cluster_selection_epsilon", 3.0
                    ),
                    cluster_allow_single_cluster=kwargs.get(
                        "step2_changepoint_allow_single_cluster", True
                    ),
                    sli_data=sli_data,
                    n_jobs=kwargs.get("step2_changepoint_n_jobs", -1),
                )
            case "changepoint-kde":
                metric_to_changepoints: dict[str, npt.ArrayLike] = {metric: res.outliers for metric, res in step1_results.items() if res.has_kept}
                if len(metric_to_changepoints) == 0:
                    return partial(
                        multireducer.change_point_clustering_with_kde,
                        search_method=kwargs["step2_changepoint_search_method"],
                        cost_model=kwargs["step2_changepoint_cost_model"],
                        penalty=kwargs.get("step2_changepoint_penalty", "aic"),
                        n_bkps=kwargs.get("step2_changepoint_n_bkps", 2),
                        kde_bandwidth=kwargs["step2_changepoint_kde_bandwidth"],
                        kde_bandwidth_adjust=kwargs["step2_changepoint_kde_bandwidth_adjust"],
                        multi_change_points=kwargs["step2_changepoint_multi_change_points"],
                        representative_method=kwargs.get("step2_changepoint_representative_method", False),
                        segment_selection_method=kwargs.get("step2_changepoint_segment_selection_method", "max"),
                        n_jobs=kwargs.get("step2_changepoint_n_jobs", -1),
                    )
                else:
                    return partial(
                        _wrap_change_point_clustering_with_kde,
                        metric_to_changepoints=metric_to_changepoints,
                        kde_bandwidth=kwargs["step2_changepoint_kde_bandwidth"],
                        kde_bandwidth_adjust=kwargs["step2_changepoint_kde_bandwidth_adjust"],
                        segment_selection_method=kwargs.get("step2_changepoint_segment_selection_method", "max"),
                    )
            case _:
                raise ValueError(
                    'method_name must be "hierarchy" or "dbscan" or "changepoint'
                )


def _wrap_change_point_clustering_with_kde(
    data: pd.DataFrame, metric_to_changepoints: dict[str, npt.ArrayLike], kde_bandwidth: float | str, kde_bandwidth_adjust: float, segment_selection_method: str,
):
    changepoints = [metric_to_changepoints[metric] for metric in data.columns]
    return multireducer.change_point_clustering_with_kde_by_changepoints(
        data, changepoints=changepoints, kde_bandwidth=kde_bandwidth, kde_bandwidth_adjust=kde_bandwidth_adjust, segment_selection_method=segment_selection_method)


def filter_out_no_change_metrics(
    data_df: pd.DataFrame, parallel: bool = False
) -> pd.DataFrame:
    vf: Callable = np.vectorize(lambda x: np.isnan(x) or x == 0)

    def filter(x: pd.Series) -> bool:
        # pd.Series.diff returns a series with the first element is NaN
        if (
            x.isna().all()
            or (x == x.iat[0]).all()
            or ((diff_x := np.diff(x)) == diff_x[0]).all()
        ):
            return False
        # remove an array including only the same value or nan
        return not vf(diff_x).all()

    if parallel:
        return data_df.loc[:, data_df.parallel_apply(filter)]
    else:
        return data_df.loc[:, data_df.apply(filter)]


def filter_out_duplicated_metrics(
    data_df: pd.DataFrame, pk: PriorKnowledge
) -> pd.DataFrame:
    def duplicated(x: pd.DataFrame) -> list[str]:
        indices = np.unique(x.T.to_numpy(), return_index=True, axis=0)[1]
        return x.iloc[:, indices].columns.tolist()

    unique_cols: list[str] = []
    for service, containers in pk.get_containers_of_service().items():
        # 1. service-level duplication
        service_metrics_df = data_df.loc[
            :, data_df.columns.str.startswith(f"s-{service}_")
        ]
        if len(service_metrics_df.columns) > 1:
            unique_cols += duplicated(service_metrics_df)
        # 2. container-level duplication
        for container in containers:
            # perform clustering in each type of metric
            # TODO: retrieve container and middleware metrics efficently
            container_metrics_df = data_df.loc[
                :,
                data_df.columns.str.startswith((f"c-{container}_", f"m-{container}_")),
            ]
            if len(container_metrics_df.columns) <= 1:
                continue
            unique_cols += duplicated(container_metrics_df)
    # 3. node-level clustering
    for node in pk.get_nodes():
        node_metrics_df = data_df.loc[:, data_df.columns.str.startswith(f"n-{node}_")]
        if len(node_metrics_df.columns) <= 1:
            continue
        unique_cols += duplicated(node_metrics_df)
    return data_df.loc[:, unique_cols]
