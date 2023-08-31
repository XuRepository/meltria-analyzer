import os
from dataclasses import dataclass

import pandas as pd

from eval import groundtruth
from meltria.priorknowledge.priorknowledge import PriorKnowledge


@dataclass
class DatasetRecord:
    """A record of dataset"""

    data_df: pd.DataFrame
    pk: PriorKnowledge
    meta: dict[str, str]
    metrics_file: str  # path of metrics file eg. '2021-12-09-argowf-chaos-hg68n-carts-db_pod-cpu-hog_4.json'

    def __hash__(self) -> int:
        return hash(self.target_app() + self.chaos_case_full())

    def target_app(self) -> str:
        """target-application eg. 'train-ticket'"""
        return self.meta["target_app"]

    def chaos_comp(self) -> str:
        """chaos-injected component eg. 'carts-db'"""
        return self.meta["chaos_injected_component"]

    def chaos_type(self) -> str:
        """injected chaos type eg. 'pod-cpu-hog'"""
        return self.meta["injected_chaos_type"]

    def grafana_dashboard_url(self) -> str:
        return self.meta["grafana_dashboard_url"]

    def chaos_case(self) -> str:
        return f"{self.chaos_comp()}/{self.chaos_type()}"

    def chaos_case_full(self) -> str:
        return f"{self.chaos_case()}/{self.chaos_case_num()}"

    def chaos_case_file(self) -> str:
        return f"{self.basename_of_metrics_file()} of {self.chaos_case()}"

    def chaos_case_num(self) -> str:
        # eg. '2021-12-09-argowf-chaos-hg68n-carts-db_pod-cpu-hog_4.json'
        return (
            self.local_dataset_id()
            + "-"
            + (
                self.basename_of_metrics_file()
                .rsplit("_", maxsplit=1)[1]
                .removesuffix(".json")
            )
        )

    def metrics_names(self) -> list[str]:
        return self.data_df.columns.tolist()  # type: ignore

    def basename_of_metrics_file(self) -> str:
        return os.path.basename(self.metrics_file)

    def local_dataset_id(self) -> str:
        # eg. '2021-12-09-argowf-chaos-hg68n-carts-db_pod-cpu-hog_4.json'
        return self.basename_of_metrics_file().split("-")[5]

    def ground_truth_metrics_frame(self) -> pd.DataFrame | None:
        _, ground_truth_metrics = groundtruth.check_tsdr_ground_truth_by_route(
            pk=self.pk,
            metrics=self.metrics_names(),  # pre-reduced data frame
            chaos_type=self.chaos_type(),
            chaos_comp=self.chaos_comp(),
        )
        if len(ground_truth_metrics) < 1:
            return None
        ground_truth_metrics.sort()
        return self.data_df[ground_truth_metrics]

    def resample_by_factor(self, factor: int):
        self.data_df = self.data_df.iloc[::factor, :]
