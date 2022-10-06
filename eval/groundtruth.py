import re
from collections import defaultdict
from functools import cache
from typing import Any, Final

import networkx as nx

import diagnoser.metric_node as mn
from meltria.metric_types import METRIC_TYPE_CONTAINERS, METRIC_TYPE_MIDDLEWARES, METRIC_TYPE_SERVICES
from meltria.priorknowledge.priorknowledge import PriorKnowledge

# TODO: define this by each target app.
CHAOS_TO_CAUSE_METRIC_PATTERNS: Final[dict[str, dict[tuple[str, str], list[str]]]] = {
    "pod-cpu-hog": {
        ("*", "container"): [
            "cpu_usage_seconds_total",
            "cpu_user_seconds_total",
            "threads",
            # "sockets",
            # "file_descriptors",
            "processes",
            # "memory_cache",
            # "memory_mapped_file",
        ],
        ("*", "jvm"): [
            "java_lang_OperatingSystem_SystemCpuLoad",
            "java_lang_OperatingSystem_ProcessCpuLoad",
            "java_lang_OperatingSystem_ProcessCpuTime",
        ],
        ("web", "jvm"): [
            "Tomcat_.+_processingTime",
            "Tomcat_.+_requestProcessingTime",
            "Tomcat_.+_requestCount",
        ],
        ("*", "mongodb"): [
            "mongodb_sys_cpu_processes",
            "mongodb_sys_cpu_procs_running",
            "mongodb_sys_cpu_user_ms",
            "mongodb_sys_cpu_idle_ms",
            "mongodb_sys_cpu_ctxt",
        ],
    },
    "pod-memory-hog": {
        ("*", "container"): [
            "memory_max_usage_byte",
            "memory_rss",
            "memory_usage_bytes",
            "memory_working_set_bytes",
            "threads",
            # "sockets",
            # "file_descriptors",
            "processes",
            # "fs_inodes_total",
            # "fs_limit_bytes",
            # "ulimits_soft",
        ],
        ("*", "jvm"): [
            "java_lang_Memory_HeapMemoryUsage_used",
            "java_lang_Memory_HeapMemoryUsage_committed",
            "java_lang_Memory_NonHeapMemoryUsage_used",
            "java_lang_Memory_NonHeapMemoryUsage_committed",
            "java_lang_MemoryPool_CollectionUsage_committed",
            "java_lang_MemoryPool_CollectionUsage_used",
            "java_lang_MemoryPool_Usage_used",
            "java_lang_MemoryPool_PeakUsage_committed",
            "java_lang_MemoryPool_Usage_committed",
            "java_lang_MemoryPool_PeakUsage_used",
            "java_lang_GarbageCollector_CollectionTime",
            "java_lang_GarbageCollector_CollectionCount",
            "java_lang_GarbageCollector_LastGcInfo_duration",
            "java_lang_GarbageCollector_LastGcInfo_memoryUsageAfterGc_init",
            "java_lang_GarbageCollector_LastGcInfo_memoryUsageAfterGc_committed",
            "java_lang_GarbageCollector_LastGcInfo_memoryUsageAfterGc_used",
            "java_lang_GarbageCollector_LastGcInfo_memoryUsageAfterGc_max",
            "java_lang_GarbageCollector_LastGcInfo_memoryUsageBeforeGc_used",
            "java_lang_GarbageCollector_LastGcInfo_memoryUsageBeforeGc_committed",
            "java_lang_GarbageCollector_LastGcInfo_memoryUsageBeforeGc_init",
            "java_lang_GarbageCollector_LastGcInfo_memoryUsageBeforeGc_max",
            "java_lang_GarbageCollector_LastGcInfo_GcThreadCount",
            "java_lang_GarbageCollector_LastGcInfo_startTime",
            "java_lang_GarbageCollector_LastGcInfo_endTime",
            "java_lang_OperatingSystem_SystemCpuLoad",
            "java_lang_OperatingSystem_FreePhysicalMemorySize",
            "java_nio_BufferPool_MemoryUsed",
            "java_nio_BufferPool_TotalCapacity",
            "java_nio_BufferPool_Count",
        ],
        ("web", "jvm"): [
            "Tomcat_.+_processingTime",
            "Tomcat_.+_requestProcessingTime",
            "Tomcat_.+_requestCount",
        ],
        ("*", "mongodb"): [
            "mongodb_sys_memory_Buffers_kb",
            "mongodb_sys_memory_MemAvailable_kb",
            "mongodb_sys_memory_MemFree_kb",
            "mongodb_sys_memory_Active_kb",
            "mongodb_sys_memory_Active_file_kb",
        ],
    },
    "pod-network-loss": {
        ("*", "container"): [
            "network_receive_bytes_total",
            "network_receive_packets_total",
            "network_transmit_bytes_total",
            "network_transmit_packets_total",
        ],
        ("*", "jvm"): [],
        ("web", "jvm"): [
            "Tomcat_.+_processingTime",
            "Tomcat_.+_requestProcessingTime",
            "Tomcat_.+_requestCount",
            "Tomcat_.+_requestBytesReceived",
            "Tomcat_.+_requestBytesSent",
            "Tomcat_.+_errorCount",
            "Tomcat_.+_bytesReceived",
            "Tomcat_.+_bytesSent",
            "Tomcat_ThreadPool_connectionCount",
            "Tomcat_ThreadPool_keepAliveCount",
        ],
        ("*", "mongodb"): [
            "mongodb_ss_network_bytesIn",
            "mongodb_ss_network_bytesOut",
            "mongodb_ss_network_physicalBytesIn",
            "mongodb_ss_network_physicalBytesOut",
            "mongodb_ss_network_numRequests",
        ],
    },
    "pod-network-latency": {
        ("*", "container"): [
            "network_receive_bytes_total",
            "network_receive_packets_total",
            "network_transmit_bytes_total",
            "network_transmit_packets_total",
        ],
        ("*", "jvm"): [],
        ("web", "jvm"): [
            "Tomcat_.+_processingTime",
            "Tomcat_.+_requestProcessingTime",
            "Tomcat_.+_requestCount",
            "Tomcat_.+_requestBytesReceived",
            "Tomcat_.+_requestBytesSent",
            "Tomcat_.+_errorCount",
            "Tomcat_.+_bytesReceived",
            "Tomcat_.+_bytesSent",
            "Tomcat_ThreadPool_connectionCount",
            "Tomcat_ThreadPool_keepAliveCount",
        ],
        ("*", "mongodb"): [
            "mongodb_ss_network_bytesIn",
            "mongodb_ss_network_bytesOut",
            "mongodb_ss_network_physicalBytesIn",
            "mongodb_ss_network_physicalBytesOut",
            "mongodb_ss_network_numRequests",
        ],
    },
}


@cache
def generate_tsdr_ground_truth(pk: PriorKnowledge) -> dict[str, Any]:
    """Generate ground truth for testing extracted metrics with tsdr based on call graph."""
    all_gt_routes: dict[str, dict[str, list[list[str]]]] = defaultdict(lambda: defaultdict(list))
    for chaos, metric_patterns_by_runtime in CHAOS_TO_CAUSE_METRIC_PATTERNS.items():
        for ctnr in pk.get_containers(skip=True):
            routes: list[list[str]] = all_gt_routes[chaos][ctnr]
            cause_service: str = pk.get_service_by_container(ctnr)
            stos_routes: list[tuple[str, ...]] = pk.get_service_routes(cause_service)

            # allow to match any of multiple routes
            for stos_route in stos_routes:
                metrics_pattern_list: list[str] = []

                role, runtime = pk.get_role_and_runtime_by_container(ctnr)

                # add cause metrics pattern
                if pk.is_target_metric_type(METRIC_TYPE_CONTAINERS):
                    for _role in ["*", role]:
                        ctnr_metric_patterns: list[str] | None = metric_patterns_by_runtime.get((_role, "container"))
                        if ctnr_metric_patterns is not None and len(ctnr_metric_patterns) > 0:
                            metrics_pattern_list.append(f"^c-{ctnr}_({'|'.join(ctnr_metric_patterns)})$")

                if pk.is_target_metric_type(METRIC_TYPE_MIDDLEWARES):
                    for _role in ["*", role]:
                        middleware_metric_patterns: list[str] | None = metric_patterns_by_runtime.get((_role, runtime))
                        if middleware_metric_patterns is not None and len(middleware_metric_patterns) > 0:
                            metrics_pattern_list.append(f"^m-{ctnr}_({'|'.join(middleware_metric_patterns)})$")

                if pk.is_target_metric_type(METRIC_TYPE_SERVICES):
                    # NOTE: duplicate service metrics are not allowed in a route
                    service_metrics_pattern: str = f"^s-{cause_service}_.+$"
                    if service_metrics_pattern not in metrics_pattern_list:
                        metrics_pattern_list.append(service_metrics_pattern)
                    if cause_service != pk.get_root_service():
                        metrics_pattern_list.append(f"^s-({'|'.join(stos_route)})_.+")

                # TODO: processing node metrics

                routes.append(metrics_pattern_list)
    return all_gt_routes


def get_tsdr_ground_truth(pk: PriorKnowledge, chaos_type: str, chaos_comp: str) -> list[list[str]]:
    return generate_tsdr_ground_truth(pk)[chaos_type][chaos_comp]


def select_ground_truth_metrics_in_routes(
    pk: PriorKnowledge,
    metrics: list[str],
    chaos_type: str,
    chaos_comp: str,
) -> list[tuple[list[str], list[str]]]:
    gt_metrics_routes: list[list[str]] = get_tsdr_ground_truth(pk, chaos_type, chaos_comp)
    candidates: list[tuple[list[str], list[str]]] = []
    for gt_route_matcher in gt_metrics_routes:
        _, match_metrics = check_route(metrics, gt_route_matcher)
        # unique insert
        if all([set(candidate[0]) != set(match_metrics) for candidate in candidates]):
            candidates.append((match_metrics, gt_route_matcher))
    return candidates


def check_tsdr_ground_truth_by_route(
    pk: PriorKnowledge, metrics: list[str], chaos_type: str, chaos_comp: str
) -> tuple[bool, list[str]]:
    gt_metrics_routes: list[list[str]] = get_tsdr_ground_truth(pk, chaos_type, chaos_comp)
    routes_ok: list[tuple[bool, list[str]]] = []
    for gt_route in gt_metrics_routes:
        ok, match_metrics = check_route(metrics, gt_route)
        routes_ok.append((ok, match_metrics))
    for ok, match_metrics in routes_ok:
        if ok:
            return True, match_metrics

    # return longest match_metrics in routes_ok
    max_len = 0
    longest_match_metrics: list[str] = []
    for _, match_metrics in routes_ok:
        if max_len < len(match_metrics):
            max_len = len(match_metrics)
            longest_match_metrics = match_metrics
    return False, longest_match_metrics


def check_route(metrics: list[str], gt_route: list[str]) -> tuple[bool, list[str]]:
    match_metrics: list[str] = []
    gt_metrics_ok = {metric: False for metric in gt_route}
    for metric in metrics:
        for metric_pattern in gt_route:
            if re.match(metric_pattern, metric):
                gt_metrics_ok[metric_pattern] = True
                match_metrics.append(metric)
    for ok in gt_metrics_ok.values():
        if not ok:
            # return partially correct metrics
            return False, match_metrics
    return True, match_metrics


def check_cause_metrics(
    pk: PriorKnowledge, nodes: mn.MetricNodes, chaos_type: str, chaos_comp: str
) -> tuple[bool, mn.MetricNodes]:
    cause_metrics: list[mn.MetricNode] = []
    for node in nodes:
        if node.is_container():
            role, _ = pk.get_role_and_runtime_by_container(node.comp)
            for _role in ["*", role]:
                patterns = CHAOS_TO_CAUSE_METRIC_PATTERNS[chaos_type].get((_role, "container"))
                if patterns is not None and len(patterns) > 0:
                    if re.match(f"^c-{chaos_comp}_({'|'.join(patterns)})$", node.label):
                        cause_metrics.append(node)
        elif node.is_node():
            patterns = CHAOS_TO_CAUSE_METRIC_PATTERNS[chaos_type].get(("*", "node"))  # FIXME: handling any other role
            if patterns is not None and len(patterns) > 0:
                if re.match(f"^n-{chaos_comp}_({'|'.join(patterns)})$", node.label):
                    cause_metrics.append(node)
        elif node.is_middleware():
            role, runtime = pk.get_role_and_runtime_by_container(node.comp)
            for _role in ["*", role]:
                patterns = CHAOS_TO_CAUSE_METRIC_PATTERNS[chaos_type].get((_role, runtime))
                if patterns is not None and len(patterns) > 0:
                    if re.match(f"^m-{chaos_comp}_({'|'.join(patterns)})$", node.label):
                        cause_metrics.append(node)
        else:
            assert False, f"Unknown metric node type: {node}"
    ret = mn.MetricNodes.from_list_of_metric_node(cause_metrics)
    if len(cause_metrics) > 0:
        return True, ret
    return False, ret


def check_causal_graph(
    pk: PriorKnowledge,
    G: nx.DiGraph,
    chaos_type: str,
    chaos_comp: str,
) -> tuple[bool, list[mn.MetricNodes]]:
    """Check that the causal graph (G) has the accurate route."""
    call_graph: nx.DiGraph = G.reverse()  # for traverse starting from root node
    cause_metric_exps: list[str] = CHAOS_TO_CAUSE_METRIC_PATTERNS[chaos_type][("*", "container")]
    cause_metric_pattern: re.Pattern = re.compile(f"^c-{chaos_comp}_({'|'.join(cause_metric_exps)})$")

    match_routes: list[mn.MetricNodes] = []
    leaves = [n for n in call_graph.nodes if n.label not in pk.get_root_metrics()]
    roots = [mn.MetricNode(r) for r in pk.get_root_metrics() if call_graph.has_node(mn.MetricNode(r))]
    for root in roots:
        for path in nx.all_simple_paths(call_graph, source=root, target=leaves):
            if len(path) <= 1:
                continue
            # compare the path with ground truth paths
            for i, node in enumerate(path[1:], start=1):  # skip ROOT_METRIC
                prev_node: mn.MetricNode = path[i - 1]
                if node.is_service():
                    if prev_node.is_container():
                        prev_service = pk.get_service_by_container(prev_node.comp)
                    else:
                        prev_service = prev_node.comp
                    if not pk.get_service_call_digraph().has_edge(prev_service, node.comp):
                        break
                elif node.is_container():
                    if prev_node.is_service():
                        cur_service = pk.get_service_by_container(node.comp)
                        if not (
                            prev_node.comp == cur_service
                            or pk.get_service_call_digraph().has_edge(prev_node.comp, cur_service),
                        ):
                            break
                    elif prev_node.is_container():
                        if not (
                            prev_node.comp == node.comp
                            or pk.get_container_call_digraph().has_edge(prev_node.comp, node.comp),
                        ):
                            break
                    if i == (len(path) - 1):  # is leaf?
                        if cause_metric_pattern.match(node.label):
                            match_routes.append(mn.MetricNodes.from_list_of_metric_node(path))
                            break
                # TODO: middleware
    return len(match_routes) > 0, match_routes
