
from copy import deepcopy

from tools.graph_refs import make_environment_ref, parse_environment_ref


PHYS_BRANCH = "phys"
CONC_BRANCH = "conc"


def get_next_environment_ref(graph: dict) -> str:
    max_id = 0
    for node in graph.get("environments_node", []):
        env_id = parse_environment_ref(node.get("id"))
        if env_id is not None:
            max_id = max(max_id, env_id)
    meta_max = parse_environment_ref(graph.get("_meta", {}).get("max_environment_ref"))
    if meta_max is not None:
        max_id = max(max_id, meta_max)
    return make_environment_ref(max_id + 1)


def _normalize_node_fields(node: dict, branch: str) -> None:
    node.setdefault("branch", branch)
    node.setdefault("parent_id", None)
    if not isinstance(node.get("children_ids"), list):
        node["children_ids"] = []
    node.setdefault("description", "")
    if branch == PHYS_BRANCH:
        node.setdefault("location", "")
        if not isinstance(node.get("minutia"), list):
            node["minutia"] = []
        node.setdefault("atmosphere", "")
    else:
        if not isinstance(node.get("attributes"), dict):
            node["attributes"] = {}


def ensure_environment_graph_ids(graph: dict | None) -> tuple[dict, bool]:
    graph = deepcopy(graph or {})
    graph.setdefault("_meta", {})
    graph.setdefault("environments_node", [])

    graph.pop("environments_relationship", None)

    changed = False
    nodes_by_id: dict[str, dict] = {}
    max_numeric_id = 0

    for node in graph["environments_node"]:
        if not isinstance(node, dict):
            continue
        env_id = node.get("id")
        parsed = parse_environment_ref(env_id)
        if parsed is None:
            parsed = max_numeric_id + 1
            node["id"] = make_environment_ref(parsed)
            changed = True
        else:
            node["id"] = make_environment_ref(parsed)
        max_numeric_id = max(max_numeric_id, parsed)
        nodes_by_id[node["id"]] = node

    phys_root_id = graph.get("phys_root_id")
    conc_root_id = graph.get("conc_root_id")
    if phys_root_id and phys_root_id not in nodes_by_id:
        phys_root_id = None
    if conc_root_id and conc_root_id not in nodes_by_id:
        conc_root_id = None

    if phys_root_id is None:
        max_numeric_id += 1
        phys_root_id = make_environment_ref(max_numeric_id)
        root_node = {
            "id": phys_root_id,
            "name": "WorldRoot:Physical",
            "branch": PHYS_BRANCH,
            "parent_id": None,
            "children_ids": [],
            "description": "Physical-space root node (auto-created).",
            "location": "",
            "minutia": [],
            "atmosphere": "",
        }
        graph["environments_node"].append(root_node)
        nodes_by_id[phys_root_id] = root_node
        graph["phys_root_id"] = phys_root_id
        changed = True
    else:
        graph["phys_root_id"] = phys_root_id

    if conc_root_id is None:
        max_numeric_id += 1
        conc_root_id = make_environment_ref(max_numeric_id)
        root_node = {
            "id": conc_root_id,
            "name": "WorldRoot:Conceptual",
            "branch": CONC_BRANCH,
            "parent_id": None,
            "children_ids": [],
            "description": "Worldbuilding / rule concept root node (auto-created).",
            "attributes": {},
        }
        graph["environments_node"].append(root_node)
        nodes_by_id[conc_root_id] = root_node
        graph["conc_root_id"] = conc_root_id
        changed = True
    else:
        graph["conc_root_id"] = conc_root_id

    for node_id, node in list(nodes_by_id.items()):
        if node_id in (graph["phys_root_id"], graph["conc_root_id"]):
            _normalize_node_fields(node, node.get("branch") or PHYS_BRANCH)
            continue
        branch = node.get("branch") or PHYS_BRANCH
        node["branch"] = branch
        _normalize_node_fields(node, branch)

        parent_id = node.get("parent_id")
        if parent_id is None or parent_id not in nodes_by_id:
            root_for_branch = graph["phys_root_id"] if branch == PHYS_BRANCH else graph["conc_root_id"]
            node["parent_id"] = root_for_branch
            changed = True

    for node in nodes_by_id.values():
        node["children_ids"] = []
    for node in nodes_by_id.values():
        parent_id = node.get("parent_id")
        if parent_id and parent_id in nodes_by_id:
            parent = nodes_by_id[parent_id]
            if node["id"] not in parent["children_ids"]:
                parent["children_ids"].append(node["id"])

    graph["_meta"]["max_environment_ref"] = make_environment_ref(max_numeric_id)
    return graph, changed


def get_environment_node_by_ref(graph: dict, environment_ref: str) -> dict | None:
    if not environment_ref:
        return None
    for node in graph.get("environments_node", []):
        if node.get("id") == environment_ref or node.get("name") == environment_ref:
            return node
    return None


def get_environment_name_by_ref(graph: dict, environment_ref: str) -> str | None:
    node = get_environment_node_by_ref(graph, environment_ref)
    return node.get("name") if node else None


def walk_ancestors(graph: dict, node_id: str) -> list[dict]:
    result: list[dict] = []
    nodes_by_id = {n.get("id"): n for n in graph.get("environments_node", [])}
    current = nodes_by_id.get(node_id)
    if not current:
        return result
    parent_id = current.get("parent_id")
    seen = set()
    while parent_id and parent_id in nodes_by_id and parent_id not in seen:
        seen.add(parent_id)
        parent = nodes_by_id[parent_id]
        result.append(parent)
        parent_id = parent.get("parent_id")
    return result


def find_lca(graph: dict, node_id_a: str, node_id_b: str) -> str | None:
    nodes_by_id = {n.get("id"): n for n in graph.get("environments_node", [])}
    a = nodes_by_id.get(node_id_a)
    b = nodes_by_id.get(node_id_b)
    if not a or not b or a.get("branch") != b.get("branch"):
        return None
    a_chain = [node_id_a] + [n["id"] for n in walk_ancestors(graph, node_id_a)]
    b_chain_set = set([node_id_b] + [n["id"] for n in walk_ancestors(graph, node_id_b)])
    for nid in a_chain:
        if nid in b_chain_set:
            return nid
    return None
