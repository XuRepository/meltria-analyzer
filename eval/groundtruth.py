import re
from functools import cache
from itertools import product
from typing import Final, cast

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
            "mongodb_ss_opLatencies_latency",
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
            "mongodb_ss_opLatencies_latency",
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
            "mongodb_ss_opLatencies_latency",
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
            "mongodb_ss_opLatencies_latency",
        ],
    },
}

NEIGHBORS_METRIC_PATTERNS_IN_SERVICE: Final[dict[str, dict[tuple[tuple[str, str], tuple[str, str]], list[str]]]] = {
    # (current component[role, runtime], neighbor component[role runtime]) -> list of neighbor metrics
    "pod-cpu-hog": {
        (("*", "container"), ("web", "jvm")): [
            "Tomcat_.+_processingTime",
            "Tomcat_.+_requestProcessingTime",
        ],
    },
    "pod-memory-hog": {
        (("*", "container"), ("web", "jvm")): [
            "Tomcat_.+_processingTime",
            "Tomcat_.+_requestProcessingTime",
        ],
        (("*", "container"), ("db", "mongodb")): [
            "mongodb_ss_opLatencies_latency",
            "mongodb_ss_opLatencies_ops",
        ],
    },
    "pod-network-loss": {
        (("*", "container"), ("*", "container")): [
            "network_receive_bytes_total",
            "network_receive_packets_total",
            "network_transmit_bytes_total",
            "network_transmit_packets_total",
        ],
        (("*", "container"), ("web", "jvm")): [
            "Tomcat_.+_processingTime",
            "Tomcat_.+_requestProcessingTime",
            "Tomcat_.+_requestCount",
            "Tomcat_.+_requestBytesReceived",
            "Tomcat_.+_requestBytesSent",
            "Tomcat_.+_bytesReceived",
            "Tomcat_.+_bytesSent",
        ],
        (("*", "container"), ("db", "mongodb")): [
            "mongodb_ss_opLatencies_latency",
            "mongodb_ss_opLatencies_ops",
            "mongodb_ss_network_bytesIn",
            "mongodb_ss_network_bytesOut",
            "mongodb_ss_network_physicalBytesIn",
            "mongodb_ss_network_physicalBytesOut",
            "mongodb_ss_network_numRequests",
        ],
    },
    "pod-network-latency": {
        (("*", "container"), ("*", "container")): [
            "network_receive_bytes_total",
            "network_receive_packets_total",
            "network_transmit_bytes_total",
            "network_transmit_packets_total",
        ],
        (("*", "container"), ("web", "jvm")): [
            "Tomcat_.+_processingTime",
            "Tomcat_.+_requestProcessingTime",
            "Tomcat_.+_requestCount",
            "Tomcat_.+_requestBytesReceived",
            "Tomcat_.+_requestBytesSent",
            "Tomcat_.+_bytesReceived",
            "Tomcat_.+_bytesSent",
        ],
        (("*", "container"), ("db", "mongodb")): [
            "mongodb_ss_opLatencies_latency",
            "mongodb_ss_opLatencies_ops",
            "mongodb_ss_network_bytesIn",
            "mongodb_ss_network_bytesOut",
            "mongodb_ss_network_physicalBytesIn",
            "mongodb_ss_network_physicalBytesOut",
            "mongodb_ss_network_numRequests",
        ],
    },
}

METRIC_PATTERNS_ON_ROUTE: Final[dict[tuple[str, str], list[str]]] = {
    ("*", "container"): [
        "network_receive_bytes_total",
        "network_receive_packets_total",
        "network_transmit_bytes_total",
        "network_transmit_packets_total",
    ],
    ("web", "jvm"): [
        "Tomcat_.+_processingTime",
        "Tomcat_.+_requestProcessingTime",
        "Tomcat_.+_requestCount",
        "Tomcat_.+_requestBytesReceived",
        "Tomcat_.+_requestBytesSent",
        "Tomcat_.+_bytesReceived",
        "Tomcat_.+_bytesSent",
    ],
}

DEFAULT_GT_OPTS: dict[str, bool] = {
    "cause_middleware": True,
    "cause_service": True,
    "neighbors_in_cause_service": True,
    "propagated_route": False,
}


def get_ground_truth_for_neighbors_in_service(
    pk: PriorKnowledge, chaos_type: str, ctnr: str
) -> dict[str, tuple[str, list[str]]]:
    role, runtime = pk.get_role_and_runtime_by_container(ctnr)
    neighbors: list[str] = pk.get_container_neighbors_in_service(ctnr)
    neighbor_metrics: dict[str, tuple[str, list[str]]] = {}
    for neighbor in neighbors:
        neighbor_role, neighbor_runtime = pk.get_role_and_runtime_by_container(neighbor)
        for _runtime, _neighbor_runtime in product(("container", runtime), ("container", neighbor_runtime)):
            for _role, _neighbor_role in [("*", "*"), (role, "*"), ("*", neighbor_role), (role, neighbor_role)]:
                _metrics = NEIGHBORS_METRIC_PATTERNS_IN_SERVICE[chaos_type].get(
                    ((_role, _runtime), (_neighbor_role, _neighbor_runtime)),
                    [],
                )
                if len(_metrics) > 0:
                    neighbor_metrics[neighbor] = (_neighbor_runtime, _metrics)
    return neighbor_metrics


def get_ground_truth_on_propagated_route(pk: PriorKnowledge, ctnr: str) -> dict[str, tuple[str, list[str]]]:
    route_metrics: dict[str, tuple[str, list[str]]] = {}
    dependent_g: nx.DiGraph = pk.get_container_call_digraph().reverse()
    for path in nx.all_simple_paths(dependent_g, source=ctnr, target=pk.get_root_container()):
        if len(path) <= 1:
            continue
        for _ctnr in path[1:]:
            role, runtime = pk.get_role_and_runtime_by_container(_ctnr)
            for (_role, _runtime) in product(["*", role], ["container", runtime]):
                if len(metrics := METRIC_PATTERNS_ON_ROUTE.get((_role, _runtime), [])) > 0:
                    route_metrics[_ctnr] = (_runtime, metrics)
    return route_metrics


def get_tsdr_ground_truth(
    pk: PriorKnowledge,
    chaos_type: str,
    chaos_comp: str,
    opts: dict[str, bool] = DEFAULT_GT_OPTS,
) -> list[list[str]]:
    """Get ground truth for testing extracted metrics with tsdr based on call graph."""
    return _get_tsdr_ground_truth(
        pk,
        chaos_type,
        chaos_comp,
        opt_cause_middleware=opts["cause_middleware"],
        opt_cause_service=opts["cause_service"],
        opt_neighbors_in_cause_service=opts["neighbors_in_cause_service"],
        opt_propagated_route=opts["propagated_route"],
    )


@cache
def _get_tsdr_ground_truth(
    pk: PriorKnowledge,
    chaos_type: str,
    chaos_comp: str,
    opt_cause_middleware: bool,
    opt_cause_service: bool,
    opt_neighbors_in_cause_service: bool,
    opt_propagated_route: bool,
) -> list[list[str]]:
    metric_patterns_by_runtime = CHAOS_TO_CAUSE_METRIC_PATTERNS[chaos_type]

    routes: list[list[str]] = []
    cause_service: str = pk.get_service_by_container(chaos_comp)
    stos_routes: list[tuple[str, ...]] = pk.get_service_routes(cause_service)

    # allow to match any of multiple routes
    for stos_route in stos_routes:
        metrics_pattern_list: list[str] = []

        role, runtime = pk.get_role_and_runtime_by_container(chaos_comp)

        # add cause metrics pattern
        if pk.is_target_metric_type(METRIC_TYPE_CONTAINERS):
            for _role in ["*", role]:
                ctnr_metric_patterns: list[str] = metric_patterns_by_runtime.get((_role, "container"), [])
                if len(ctnr_metric_patterns) > 0:
                    metrics_pattern_list.append(f"^c-{chaos_comp}_({'|'.join(ctnr_metric_patterns)})$")

        if pk.is_target_metric_type(METRIC_TYPE_MIDDLEWARES) and opt_cause_middleware:
            for _role in ["*", role]:
                middleware_metric_patterns: list[str] = metric_patterns_by_runtime.get((_role, runtime), [])
                if len(middleware_metric_patterns) > 0:
                    metrics_pattern_list.append(f"^m-{chaos_comp}_({'|'.join(middleware_metric_patterns)})$")

        if pk.is_target_metric_type(METRIC_TYPE_SERVICES) and opt_cause_service:
            # NOTE: duplicate service metrics are not allowed in a route
            service_metrics_pattern: str = f"^s-{cause_service}_.+$"
            if service_metrics_pattern not in metrics_pattern_list:
                metrics_pattern_list.append(service_metrics_pattern)
            if cause_service != pk.get_root_service():
                metrics_pattern_list.append(f"^s-({'|'.join(stos_route)})_.+")

        # add neighbor metrics pattern
        neighbor_metrics_with_runtime = (
            get_ground_truth_for_neighbors_in_service(pk, chaos_type, chaos_comp)
            if opt_neighbors_in_cause_service
            else {}
        )
        # add metrics pattern on fault propageted routes
        propagated_metrics_with_runtime = (
            get_ground_truth_on_propagated_route(pk, chaos_comp) if opt_propagated_route else {}
        )
        for _metrics_with_runtime in [neighbor_metrics_with_runtime, propagated_metrics_with_runtime]:
            for _ctnr, (_runtime, _metrics) in _metrics_with_runtime.items():
                match _runtime:
                    case "container":
                        if pk.is_target_metric_type(METRIC_TYPE_CONTAINERS):
                            metrics_pattern_list.append(f"^c-{_ctnr}_({'|'.join(_metrics)})$")
                    case _:
                        if pk.is_target_metric_type(METRIC_TYPE_MIDDLEWARES):
                            metrics_pattern_list.append(f"^m-{_ctnr}_({'|'.join(_metrics)})$")

        routes.append(metrics_pattern_list)
    return routes


def select_ground_truth_metrics_in_routes(
    pk: PriorKnowledge,
    metrics: list[str],
    chaos_type: str,
    chaos_comp: str,
    gt_opts: dict[str, bool] = DEFAULT_GT_OPTS,
) -> list[tuple[list[str], list[str]]]:
    gt_metrics_routes: list[list[str]] = get_tsdr_ground_truth(pk, chaos_type, chaos_comp, gt_opts)
    candidates: list[tuple[list[str], list[str]]] = []
    for gt_route_matcher in gt_metrics_routes:
        _, match_metrics = check_route(metrics, gt_route_matcher)
        # unique insert
        if all([set(candidate[0]) != set(match_metrics) for candidate in candidates]):
            candidates.append((match_metrics, gt_route_matcher))
    return candidates


def check_tsdr_ground_truth_by_route(
    pk: PriorKnowledge,
    metrics: list[str],
    chaos_type: str,
    chaos_comp: str,
    gt_opts: dict[str, bool] = DEFAULT_GT_OPTS,
) -> tuple[bool, list[str]]:
    gt_metrics_routes: list[list[str]] = get_tsdr_ground_truth(pk, chaos_type, chaos_comp, gt_opts)
    routes_ok: list[tuple[bool, list[str]]] = []
    for gt_route in gt_metrics_routes:
        ok, match_metrics = check_route(metrics, gt_route)
        routes_ok.append((ok, match_metrics))
    for ok, metrics in routes_ok:
        if ok:
            return True, metrics

    # return longest match_metrics in routes_ok
    max_len = 0
    longest_match_metrics: list[str] = []
    for _, metrics in routes_ok:
        if max_len < len(metrics):
            max_len = len(metrics)
            longest_match_metrics = metrics
    return False, longest_match_metrics


def check_route(metrics: list[str], gt_route: list[str]) -> tuple[bool, list[str]]:
    match_metrics: set[str] = set()
    gt_metrics_ok = {metric: False for metric in gt_route}
    for metric in metrics:
        for metric_pattern in gt_route:
            if re.match(metric_pattern, metric):
                gt_metrics_ok[metric_pattern] = True
                match_metrics.add(metric)
    for ok in gt_metrics_ok.values():
        if not ok:
            # return partially correct metrics
            return False, list(match_metrics)
    return True, list(match_metrics)


def check_cause_metrics(
    pk: PriorKnowledge, metrics: mn.MetricNodes | list[str], chaos_type: str, chaos_comp: str
) -> tuple[bool, mn.MetricNodes]:
    nodes: mn.MetricNodes
    if type(metrics) == list:
        nodes = mn.MetricNodes.from_metric_names(metrics)
    else:
        nodes = cast(mn.MetricNodes, metrics)
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
        elif node.is_service():
            pass
        else:
            assert False, f"Unknown metric node type: {node}"
    return len(cause_metrics) > 0, mn.MetricNodes.from_list_of_metric_node(cause_metrics)


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
