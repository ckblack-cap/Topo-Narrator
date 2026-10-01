
import os
import threading
from copy import deepcopy
from functools import wraps

from data import (
    characters_graph_data,
    raw_environments_graph_data,
    plots_graph_data,
    set_characters_graph_data,
    set_environments_graph_data,
    set_plots_graph_data,
    data_path,
    get_file,
    set_file,
)
from tools.environment_ids import (
    CONC_BRANCH,
    PHYS_BRANCH,
    ensure_environment_graph_ids,
    get_environment_node_by_ref,
    walk_ancestors,
)
from tools.event_schema import extract_active_reference_set
from tools.graph_refs import (
    is_environment_ref,
    is_real_character_ref,
    is_real_environment_ref,
    is_real_plot_ref,
    is_temp_environment_ref,
    make_character_ref,
    make_environment_ref,
    make_plot_ref,
    parse_character_ref,
    parse_environment_ref,
    parse_plot_ref,
)


def _seed_environment_parent_ref(item: dict):
    """Read one canonical seed parent edge, rejecting contradictory aliases."""
    if not isinstance(item, dict):
        raise ValueError("Seed environment must be an object.")

    def _clean(value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value.strip() if isinstance(value, str) else value

    has_parent_id = "parent_id" in item
    has_parent_hint = "parent_hint" in item
    parent_id = _clean(item.get("parent_id"))
    parent_hint = _clean(item.get("parent_hint"))
    if has_parent_id and has_parent_hint and parent_id != parent_hint:
        raise ValueError(
            f"Seed environment {item.get('temp_id')} has contradictory "
            f"parent_id={parent_id!r} and parent_hint={parent_hint!r}."
        )
    if has_parent_id:
        return parent_id
    if has_parent_hint:
        return parent_hint

    # Backward-compatible read for older Event payloads.  New proposal-backed
    # nodes always carry the canonical edge at the seed-node top level.
    profile = item.get("profile") if isinstance(item.get("profile"), dict) else {}
    return _clean(profile.get("parent_id") or profile.get("parent_hint"))


_PREMISE_LOCKS: dict[str, threading.RLock] = {}
_PREMISE_LOCKS_GUARD = threading.Lock()


def _get_premise_lock(premise_id: str) -> threading.RLock:
    """Return the in-process transaction lock for one global premise."""
    key = str(premise_id)
    with _PREMISE_LOCKS_GUARD:
        lock = _PREMISE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PREMISE_LOCKS[key] = lock
        return lock


def _locked_by_premise(function):
    """Serialize extract/sync ID allocation for the same premise in one process."""
    @wraps(function)
    def wrapper(premise_id: str, *args, **kwargs):
        with _get_premise_lock(premise_id):
            return function(premise_id, *args, **kwargs)
    return wrapper


def make_work_premise_id(premise_id: str, event_index: int) -> str:
    return f"{premise_id}__work__{int(event_index):02d}"


def _work_meta_path(work_premise_id: str) -> str:
    return os.path.join(data_path(), work_premise_id, "working_meta.json")


def _require_graph_object(value, label: str) -> dict:
    """Distinguish an unreadable graph (`False`) from a valid empty JSON object."""
    if not isinstance(value, dict):
        raise OSError(
            f"Failed to read {label}: expected a JSON object, got {type(value).__name__}."
        )
    return deepcopy(value)


def _normalize_character_graph(value, label: str) -> dict:
    graph = _require_graph_object(value, label)
    graph.setdefault("characters_node", [])
    graph.setdefault("characters_relationship", [])
    if not isinstance(graph["characters_node"], list) or not isinstance(
        graph["characters_relationship"], list
    ):
        raise ValueError(f"{label} has an invalid character graph schema.")
    return graph


def _normalize_plot_graph(value, label: str) -> dict:
    graph = _require_graph_object(value, label)
    graph.setdefault("plots_node", [])
    graph.setdefault("plots_relationship", [])
    if not isinstance(graph["plots_node"], list) or not isinstance(
        graph["plots_relationship"], list
    ):
        raise ValueError(f"{label} has an invalid plot graph schema.")
    return graph


def _work_storage_paths(work_premise_id: str) -> dict[str, str]:
    root = os.path.abspath(data_path())
    work_dir = os.path.abspath(os.path.join(root, str(work_premise_id)))
    if os.path.commonpath((root, work_dir)) != root:
        raise ValueError(f"Unsafe work premise path: {work_premise_id!r}")
    return {
        "G_C": os.path.join(work_dir, "characters.json"),
        "T_E": os.path.join(work_dir, "environments.json"),
        "C_P": os.path.join(work_dir, "plots.json"),
        "working_meta": os.path.join(work_dir, "working_meta.json"),
    }


def _snapshot_json_files(paths: dict[str, str]) -> dict[str, tuple[bool, object]]:
    snapshots: dict[str, tuple[bool, object]] = {}
    for label, path in paths.items():
        existed = os.path.exists(path)
        value = None
        if existed:
            value = get_file(path)
            if value is False:
                raise OSError(f"Failed to snapshot existing work file {label}: {path}")
        snapshots[path] = (existed, deepcopy(value))
    return snapshots


def _restore_json_file_snapshots(
    snapshots: dict[str, tuple[bool, object]],
) -> None:
    """Restore overwritten work files and delete files created by this attempt."""
    failures: list[str] = []
    parent_dirs: set[str] = set()
    for path, (existed, value) in snapshots.items():
        parent_dirs.add(os.path.dirname(path))
        if existed:
            if not set_file(path, value):
                failures.append(path)
        elif os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                failures.append(path)
    for directory in parent_dirs:
        try:
            os.rmdir(directory)
        except OSError:
            # Pre-existing or non-empty work directories are intentionally retained.
            pass
    if failures:
        raise OSError(f"Failed to restore work files: {', '.join(failures)}")


def _extract_nodes_by_ids(nodes: list[dict], ids: set) -> list[dict]:
    selected = []
    for node in nodes:
        if node.get("id") in ids:
            selected.append(deepcopy(node))
    return selected


def _current_max_plot_id(graph: dict) -> int:
    return max([int(node.get("id", 0)) for node in graph.get("plots_node", [])] or [0])


def _environment_projection_ids(graph: dict, referenced_ids: set[str]) -> set[str]:
    """Return the dual-tree roots plus every referenced node's ancestor path.

    `set_environments_graph_data` deliberately repairs malformed trees.  Passing a
    subgraph whose root ids are not backed by real root nodes therefore creates new
    roots and can later corrupt the global tree during merge.  Keeping both genuine
    roots in every M_sub also makes the work graph a valid dual-tree on its own.
    """
    nodes_by_id = {
        str(node.get("id")): node
        for node in graph.get("environments_node", [])
        if isinstance(node, dict) and node.get("id")
    }
    selected: set[str] = set()
    for root_key in ("phys_root_id", "conc_root_id"):
        root_id = graph.get(root_key)
        if not root_id or root_id not in nodes_by_id:
            raise ValueError(f"Global T_E is missing a valid {root_key}: {root_id!r}")
        selected.add(root_id)

    for environment_id in referenced_ids:
        if environment_id not in nodes_by_id:
            raise ValueError(f"R_active references an unknown environment: {environment_id}")
        selected.add(environment_id)
        selected.update(
            str(node["id"])
            for node in walk_ancestors(graph, environment_id)
            if node.get("id")
        )
    return selected


def _validate_environment_projection(graph: dict, expected_roots: dict | None = None) -> None:
    """Fail closed when a work T_E is not a self-contained dual-branch tree."""
    nodes = [node for node in graph.get("environments_node", []) if isinstance(node, dict)]
    nodes_by_id = {node.get("id"): node for node in nodes if node.get("id")}
    if len(nodes_by_id) != len(nodes):
        raise ValueError("M_sub T_E contains missing or duplicate environment ids.")

    roots = {
        "phys_root_id": (graph.get("phys_root_id"), PHYS_BRANCH),
        "conc_root_id": (graph.get("conc_root_id"), CONC_BRANCH),
    }
    reserved_root_names = {
        "WorldRoot:Physical": roots["phys_root_id"][0],
        "WorldRoot:Conceptual": roots["conc_root_id"][0],
    }
    for node_id, node in nodes_by_id.items():
        expected_root_id = reserved_root_names.get(node.get("name"))
        if expected_root_id is not None and node_id != expected_root_id:
            raise ValueError(
                f"T_E contains a duplicate reserved root name at {node_id}; "
                "the existing memory must be repaired before it can be merged safely."
            )
    for root_key, (root_id, branch) in roots.items():
        if expected_roots and root_id != expected_roots.get(root_key):
            raise ValueError(
                f"M_sub {root_key}={root_id!r} differs from global "
                f"{expected_roots.get(root_key)!r}."
            )
        root = nodes_by_id.get(root_id)
        if not root or root.get("branch") != branch or root.get("parent_id") is not None:
            raise ValueError(f"M_sub has an invalid {root_key}: {root_id!r}")

    for node_id, node in nodes_by_id.items():
        if node_id in {roots["phys_root_id"][0], roots["conc_root_id"][0]}:
            continue
        parent = nodes_by_id.get(node.get("parent_id"))
        if not parent:
            raise ValueError(f"Environment {node_id} has no parent inside M_sub.")
        if parent.get("branch") != node.get("branch"):
            raise ValueError(f"Environment {node_id} crosses T_E branches.")
        seen = {node_id}
        current = node
        while current.get("parent_id") is not None:
            parent_id = current.get("parent_id")
            if parent_id in seen:
                raise ValueError(f"Environment cycle detected at {node_id}.")
            seen.add(parent_id)
            current = nodes_by_id.get(parent_id)
            if current is None:
                raise ValueError(f"Environment {node_id} has a broken ancestor path.")


def validate_environment_projection(graph: dict, expected_roots: dict | None = None) -> None:
    """Public fail-closed validator shared by orchestration/read paths."""
    _validate_environment_projection(graph, expected_roots)


def _load_environment_graph(
    premise_id: str,
    label: str,
    *,
    initialize_pristine_global: bool = False,
) -> tuple[dict, dict]:
    """Read raw T_E, validate before normalization, then return (raw, normalized).

    A newly created premise starts as the exact empty template
    ``{"environments_node": []}``; this one state may be initialized explicitly.
    Every non-empty tree is validated before ``ensure_environment_graph_ids`` can
    repair it, so broken historical memory is never silently rewritten here.
    """
    raw = _require_graph_object(raw_environments_graph_data(premise_id), label)
    nodes = raw.get("environments_node")
    if not isinstance(nodes, list):
        raise ValueError(f"{label} has an invalid environments_node field.")

    is_pristine = (
        not nodes
        and not raw.get("phys_root_id")
        and not raw.get("conc_root_id")
    )
    if is_pristine:
        if not initialize_pristine_global:
            raise ValueError(f"{label} is missing its physical and conceptual roots.")
        normalized, _ = ensure_environment_graph_ids(raw)
        _validate_environment_projection(normalized)
        if not set_environments_graph_data(premise_id, normalized):
            raise OSError(f"Failed to initialize pristine global T_E for {premise_id}.")
        return raw, normalized

    _validate_environment_projection(raw)
    normalized, _ = ensure_environment_graph_ids(raw)
    _validate_environment_projection(
        normalized,
        {
            "phys_root_id": raw.get("phys_root_id"),
            "conc_root_id": raw.get("conc_root_id"),
        },
    )
    return raw, normalized


def _latest_plot_id(graph: dict) -> int | None:
    """Return the chronological tail used to attach the next reverse-canon node."""
    nodes = [node for node in graph.get("plots_node", []) if isinstance(node, dict) and node.get("id") is not None]
    if not nodes:
        return None
    return int(max(nodes, key=lambda node: (int(node.get("chapter", 0) or 0), int(node["id"])))["id"])


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@_locked_by_premise
def extract_working_substructure(premise_id: str, event_payload: dict, event_index: int) -> str:
    event_meta = event_payload.get("event_meta", {})
    event_id = event_meta.get("event_id", f"event_{int(event_index):02d}")
    work_premise_id = make_work_premise_id(premise_id, event_index)

    r_active = extract_active_reference_set(event_payload)
    event_payload["R_active"] = r_active

    global_characters = _normalize_character_graph(
        characters_graph_data(premise_id), "global G_C"
    )
    _, global_environments = _load_environment_graph(
        premise_id,
        "global T_E",
        initialize_pristine_global=True,
    )
    global_plots = _normalize_plot_graph(plots_graph_data(premise_id), "global C_P")

    seed_subgraph = event_payload.get("seed_subgraph", {})
    seed_characters = seed_subgraph.get("characters", {})
    seed_environments = seed_subgraph.get("environments", {})

    char_real_ids = {
        parse_character_ref(ref)
        for ref in r_active["characters"]
        if is_real_character_ref(ref)
    }
    char_real_ids = {cid for cid in char_real_ids if cid is not None}
    selected_characters = _extract_nodes_by_ids(
        global_characters.get("characters_node", []), char_real_ids
    )
    selected_character_ids = {int(node["id"]) for node in selected_characters if "id" in node}
    missing_character_ids = char_real_ids - selected_character_ids
    if missing_character_ids:
        missing_refs = ", ".join(
            make_character_ref(cid) for cid in sorted(missing_character_ids)
        )
        raise ValueError(
            f"R_active references unknown global character node(s): {missing_refs}"
        )
    selected_character_relationships = [
        deepcopy(rel)
        for rel in global_characters.get("characters_relationship", [])
        if int(rel.get("source", -1)) in selected_character_ids
        and int(rel.get("target", -1)) in selected_character_ids
    ]

    global_char_max = max(
        [int(node.get("id", 0)) for node in global_characters.get("characters_node", [])] or [0]
    )
    next_character_id = global_char_max
    character_temp_id_map: dict[str, int] = {}
    for item in seed_characters.get("new", []):
        if not isinstance(item, dict):
            continue
        profile_raw = item.get("profile", {})
        if isinstance(profile_raw, dict):
            profile = deepcopy(profile_raw)
        else:
            print(
                f"⚠️ [WorkingMemory] seed_character {item.get('temp_id')}  has a profile "
                f"that is not a dict (type={type(profile_raw).__name__}); using an empty dict instead",
                flush=True,
            )
            profile = {"description": profile_raw} if isinstance(profile_raw, str) else {}
        next_character_id += 1
        temp_id = item["temp_id"]
        character_temp_id_map[temp_id] = next_character_id
        item_aliases = item.get("aliases", [])
        aliases_clean = [
            str(a).strip()
            for a in (item_aliases if isinstance(item_aliases, list) else [])
            if isinstance(a, (str, int)) and str(a).strip()
        ]
        selected_characters.append({
            "id": next_character_id,
            "name": item.get("name", temp_id),
            "aliases": aliases_clean,
            "description": profile.get("description", ""),
            "short_term_goal": profile.get("short_term", ""),
            "long_term_goal": profile.get("long_term", ""),
            "importance": profile.get("importance", "Supporting"),
            "current_plot_participation": "key character",
            "status": "initial state",
        })

    # Event-entry relationships involving newly proposed characters belong to the
    # projection itself. Relationship changes that happen in prose are still added
    # later by the Character Manager hot update.
    projected_character_ids = {
        int(node["id"])
        for node in selected_characters
        if isinstance(node, dict) and node.get("id") is not None
    }
    projected_relationship_keys = {
        (int(rel["source"]), int(rel["target"]))
        for rel in selected_character_relationships
        if isinstance(rel, dict) and rel.get("source") is not None and rel.get("target") is not None
    }

    def resolve_projected_character_ref(ref) -> int | None:
        if isinstance(ref, str) and ref in character_temp_id_map:
            return int(character_temp_id_map[ref])
        if is_real_character_ref(ref):
            return parse_character_ref(ref)
        return None

    for relationship in seed_characters.get("relationships", []):
        if not isinstance(relationship, dict):
            continue
        source_id = resolve_projected_character_ref(relationship.get("source"))
        target_id = resolve_projected_character_ref(relationship.get("target"))
        if (
            source_id is None
            or target_id is None
            or source_id == target_id
            or source_id not in projected_character_ids
            or target_id not in projected_character_ids
        ):
            raise ValueError(f"Invalid proposed character relationship: {relationship}")
        edge_key = (source_id, target_id)
        if edge_key in projected_relationship_keys:
            continue
        selected_character_relationships.append({
            "source": source_id,
            "target": target_id,
            "current_type": relationship.get("current_type", "acquaintances"),
            "change_history": deepcopy(relationship.get("change_history", [])),
        })
        projected_relationship_keys.add(edge_key)

    seed_environment_items = [
        item for item in seed_environments.get("new", []) if isinstance(item, dict)
    ]
    global_environment_nodes_by_id = {
        node.get("id"): node
        for node in global_environments.get("environments_node", [])
        if isinstance(node, dict) and node.get("id")
    }
    seed_environment_specs: dict[str, dict] = {}
    for item in seed_environment_items:
        temp_id = item.get("temp_id")
        if not is_temp_environment_ref(temp_id):
            raise ValueError(
                f"Seed environment temp_id must be canonical e_tmp_NN: {temp_id!r}"
            )
        if temp_id in seed_environment_specs:
            raise ValueError(f"Duplicate seed environment temp_id: {temp_id}")
        branch = str(item.get("branch") or "").strip().lower()
        if branch not in (PHYS_BRANCH, CONC_BRANCH):
            raise ValueError(
                f"Seed environment {temp_id} branch must be 'phys' or 'conc'."
            )
        parent_ref = _seed_environment_parent_ref(item)
        if parent_ref is not None and not is_environment_ref(parent_ref):
            raise ValueError(
                f"Seed environment {temp_id} parent must be null, e_K, or "
                f"e_tmp_NN; got {parent_ref!r}."
            )
        seed_environment_specs[temp_id] = {
            "branch": branch,
            "parent_ref": parent_ref,
        }

    for temp_id, spec in seed_environment_specs.items():
        parent_ref = spec["parent_ref"]
        if is_temp_environment_ref(parent_ref):
            parent_spec = seed_environment_specs.get(parent_ref)
            if parent_spec is None:
                raise ValueError(
                    f"Seed environment {temp_id} has unresolved temp parent {parent_ref}."
                )
            if parent_spec["branch"] != spec["branch"]:
                raise ValueError(
                    f"Seed environment {temp_id} ({spec['branch']}) cannot mount "
                    f"under {parent_ref} ({parent_spec['branch']}); branches differ."
                )
        elif is_real_environment_ref(parent_ref):
            parent_node = global_environment_nodes_by_id.get(parent_ref)
            if parent_node is None:
                raise ValueError(
                    f"Seed environment {temp_id} has unknown global parent {parent_ref}."
                )
            if parent_node.get("branch") != spec["branch"]:
                raise ValueError(
                    f"Seed environment {temp_id} ({spec['branch']}) cannot mount "
                    f"under {parent_ref} ({parent_node.get('branch')}); branches differ."
                )

    for temp_id in seed_environment_specs:
        seen = {temp_id}
        current = seed_environment_specs[temp_id]["parent_ref"]
        while is_temp_environment_ref(current):
            if current in seen:
                raise ValueError(
                    f"Seed environment temp parent chain contains a cycle at {current}."
                )
            seen.add(current)
            current = seed_environment_specs[current]["parent_ref"]

    env_real_ids = {
        ref for ref in r_active["environments"] if is_real_environment_ref(ref)
    }
    for spec in seed_environment_specs.values():
        parent_ref = spec["parent_ref"]
        if is_real_environment_ref(parent_ref):
            env_real_ids.add(parent_ref)

    projection_ids = _environment_projection_ids(global_environments, env_real_ids)
    selected_environments = _extract_nodes_by_ids(
        global_environments.get("environments_node", []), projection_ids
    )

    global_env_max = max(
        [
            parse_environment_ref(node.get("id")) or 0
            for node in global_environments.get("environments_node", [])
            if isinstance(node, dict)
        ]
        or [0]
    )
    environment_temp_id_map: dict[str, str] = {}
    for offset, item in enumerate(seed_environment_items, start=1):
        temp_id = item.get("temp_id")
        if not temp_id:
            continue
        environment_temp_id_map[temp_id] = make_environment_ref(global_env_max + offset)

    selected_environment_ids = {
        node.get("id") for node in selected_environments if isinstance(node, dict)
    }
    for item in seed_environment_items:
        if not isinstance(item, dict):
            continue
        profile_raw = item.get("profile", {})
        if isinstance(profile_raw, dict):
            profile = deepcopy(profile_raw)
        else:
            print(
                f"⚠️ [WorkingMemory] seed_environment {item.get('temp_id')}  has a profile "
                f"that is not a dict (type={type(profile_raw).__name__}); using an empty dict instead",
                flush=True,
            )
            profile = {"description": profile_raw} if isinstance(profile_raw, str) else {}
        temp_id = item["temp_id"]
        work_environment_id = environment_temp_id_map[temp_id]
        branch = seed_environment_specs[temp_id]["branch"]
        parent_ref = seed_environment_specs[temp_id]["parent_ref"]
        parent_id = environment_temp_id_map.get(parent_ref, parent_ref)
        root_id = (
            global_environments.get("phys_root_id")
            if branch == PHYS_BRANCH
            else global_environments.get("conc_root_id")
        )
        if parent_id is None:
            parent_id = root_id
        elif parent_id not in selected_environment_ids and parent_id not in environment_temp_id_map.values():
            # Contract validation above should make this unreachable.  Keep a
            # fail-closed guard here so an unresolved temp edge can never be
            # silently remounted at the branch root.
            raise ValueError(
                f"Seed environment {temp_id} parent {parent_ref} cannot be resolved "
                "inside the projected T_E."
            )
        node = {
            "id": work_environment_id,
            "name": item.get("name", temp_id),
            "branch": branch,
            "parent_id": parent_id or root_id,
            "children_ids": [],
            "description": profile.get("description", profile.get("macro_description", "")),
        }
        if branch == "phys":
            node["location"] = profile.get("location", "")
            node["minutia"] = list(profile.get("minutia", []) or (
                [profile.get("micro_details")] if profile.get("micro_details") else []
            ))
            node["atmosphere"] = profile.get("atmosphere", "")
        else:
            node["attributes"] = profile.get("attributes", {})
        selected_environments.append(node)
        selected_environment_ids.add(work_environment_id)

    requested_plot_ids = {
        parse_plot_ref(ref) for ref in r_active["plots"] if is_real_plot_ref(ref)
    }
    requested_plot_ids = {pid for pid in requested_plot_ids if pid is not None}
    plot_real_ids = set(requested_plot_ids)
    temporal_tail_id = _latest_plot_id(global_plots)
    if temporal_tail_id is not None:
        plot_real_ids.add(temporal_tail_id)
    selected_plots = _extract_nodes_by_ids(global_plots.get("plots_node", []), plot_real_ids)
    selected_plot_id_set = {int(node["id"]) for node in selected_plots if "id" in node}
    missing_plot_ids = requested_plot_ids - selected_plot_id_set
    if missing_plot_ids:
        missing_refs = ", ".join(make_plot_ref(pid) for pid in sorted(missing_plot_ids))
        raise ValueError(f"R_active references unknown global plot node(s): {missing_refs}")
    selected_plot_relationships = [
        deepcopy(rel)
        for rel in global_plots.get("plots_relationship", [])
        if int(rel[0]) in selected_plot_id_set and int(rel[1]) in selected_plot_id_set
    ]
    global_plot_max = _current_max_plot_id(global_plots)

    work_characters = {
        "_meta": {
            "id_floor": global_char_max,
            "temp_to_id": character_temp_id_map,
            "source_premise_id": premise_id,
            "event_id": event_id,
            "R_active_chars": r_active["characters"],
        },
        "characters_node": selected_characters,
        "characters_relationship": selected_character_relationships,
    }
    work_environments = {
        "_meta": {
            "temp_to_id": environment_temp_id_map,
            "source_premise_id": premise_id,
            "event_id": event_id,
            "R_active_envs": r_active["environments"],
        },
        "phys_root_id": global_environments.get("phys_root_id"),
        "conc_root_id": global_environments.get("conc_root_id"),
        "environments_node": selected_environments,
    }
    work_plots = {
        "_meta": {
            "id_floor": global_plot_max,
            "source_premise_id": premise_id,
            "event_id": event_id,
            "R_active_plots": r_active["plots"],
            "last_canon_plot_id": temporal_tail_id,
        },
        "plots_node": selected_plots,
        "plots_relationship": selected_plot_relationships,
        "canon_plot_layer": [],  # trace-only
    }

    work_environments, _ = ensure_environment_graph_ids(work_environments)
    _validate_environment_projection(
        work_environments,
        {
            "phys_root_id": global_environments.get("phys_root_id"),
            "conc_root_id": global_environments.get("conc_root_id"),
        },
    )

    work_file_snapshots = _snapshot_json_files(_work_storage_paths(work_premise_id))
    try:
        if not set_characters_graph_data(work_premise_id, work_characters):
            raise OSError(f"Failed to persist working G_C for {work_premise_id}.")
        if not set_environments_graph_data(work_premise_id, work_environments):
            raise OSError(f"Failed to persist working T_E for {work_premise_id}.")
        if not set_plots_graph_data(work_premise_id, work_plots):
            raise OSError(f"Failed to persist working C_P for {work_premise_id}.")
        if not set_file(_work_meta_path(work_premise_id), {
            "premise_id": premise_id,
            "event_id": event_id,
            "R_active": r_active,
            "character_temp_to_id": character_temp_id_map,
            "environment_temp_to_id": environment_temp_id_map,
        }):
            raise OSError(f"Failed to persist working metadata for {work_premise_id}.")
    except Exception as persist_error:
        try:
            _restore_json_file_snapshots(work_file_snapshots)
        except Exception as rollback_error:
            raise OSError(
                f"Working-memory initialization failed and rollback also failed: {rollback_error}"
            ) from persist_error
        raise
    return work_premise_id


def get_work_meta(work_premise_id: str, *, required: bool = False) -> dict:
    value = get_file(_work_meta_path(work_premise_id))
    if isinstance(value, dict):
        return value
    if required:
        raise OSError(f"Failed to read working metadata for {work_premise_id}.")
    return {}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def apply_plot_canon_patch(work_premise_id: str, chapter_payload: dict, patch: dict) -> dict:
    from tools.plot_graph_manager import normalize_edge_type

    work_plots = _normalize_plot_graph(
        plots_graph_data(work_premise_id), f"working C_P ({work_premise_id})"
    )
    _, work_environments = _load_environment_graph(
        work_premise_id, f"working T_E ({work_premise_id})"
    )
    work_meta = get_work_meta(work_premise_id)

    def resolve_physical_scene(value) -> str | None:
        if isinstance(value, list):
            value = value[0] if len(value) == 1 else None
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            candidate = work_meta.get("environment_temp_to_id", {}).get(candidate, candidate)
            environment = get_environment_node_by_ref(work_environments, candidate)
            if environment and environment.get("branch") == PHYS_BRANCH:
                return str(environment.get("name") or environment.get("id"))
        return None

    fallback_scene = None
    for environment_ref in chapter_payload.get("refs", {}).get("environments", []):
        fallback_scene = resolve_physical_scene(environment_ref)
        if fallback_scene:
            break

    coverage_evidence_by_alias: dict[str, list[dict]] = {}
    coverage_contract = patch.get("plan_coverage_contract")
    coverage_items = patch.get("plan_coverage")
    if coverage_contract is not None:
        if coverage_contract != 1 or not isinstance(coverage_items, list):
            raise ValueError("Unsupported or malformed plot plan coverage contract.")
        valid_aliases = {
            f"new_node_{index}" for index, _ in enumerate(patch.get("plots_node", []))
        }
        seen_requirements: set[str] = set()
        for index, item in enumerate(coverage_items):
            if not isinstance(item, dict):
                raise ValueError(f"plan_coverage[{index}] must be an object.")
            requirement_id = item.get("requirement_id")
            source = item.get("source")
            status = item.get("status")
            evidence = item.get("evidence", [])
            covered_by = item.get("covered_by", [])
            if not isinstance(requirement_id, str) or not requirement_id.strip():
                raise ValueError(f"plan_coverage[{index}] has no requirement_id.")
            if requirement_id in seen_requirements:
                raise ValueError(f"Duplicate plan coverage requirement: {requirement_id}.")
            seen_requirements.add(requirement_id)
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"plan_coverage[{index}] has no source.")
            if not isinstance(evidence, list) or not isinstance(covered_by, list):
                raise ValueError(
                    f"plan_coverage[{index}] evidence and covered_by must be arrays."
                )
            if status == "realized_missing":
                raise ValueError(
                    f"Cannot apply reverse canon with uncovered realized beat {requirement_id}."
                )
            if status == "not_realized":
                if evidence or covered_by:
                    raise ValueError(
                        f"not_realized coverage {requirement_id} may not carry canon evidence."
                    )
                continue
            if status != "realized_covered" or not evidence or not covered_by:
                raise ValueError(f"Malformed realized coverage {requirement_id}.")
            if any(
                not isinstance(span, str) or not span.strip()
                for span in evidence
            ):
                raise ValueError(f"Coverage {requirement_id} has malformed prose evidence.")
            record = {
                "requirement_id": requirement_id,
                "source": source,
                "prose_evidence": list(evidence),
            }
            for alias in covered_by:
                if alias not in valid_aliases:
                    raise ValueError(
                        f"Coverage {requirement_id} references unknown node alias {alias!r}."
                    )
                records = coverage_evidence_by_alias.setdefault(alias, [])
                if record not in records:
                    records.append(deepcopy(record))

    id_floor = int(work_plots.get("_meta", {}).get("id_floor", 0) or 0)
    current_max_id = max(_current_max_plot_id(work_plots), id_floor)
    local_id_map: dict[str, int] = {}
    created_nodes = []

    for index, node in enumerate(patch.get("plots_node", [])):
        if not isinstance(node, dict):
            continue
        current_max_id += 1
        node_copy = deepcopy(node)
        node_copy["id"] = current_max_id
        if "scene" in node_copy and "environment" not in node_copy:
            node_copy["environment"] = node_copy.pop("scene")
        node_copy["environment"] = resolve_physical_scene(node_copy.get("environment")) or fallback_scene
        if not node_copy.get("environment"):
            raise ValueError(
                f"Canon plot node {index} has no physical scene represented in the current M_sub."
            )
        node_copy.setdefault("chapter", chapter_payload.get("chapter_no"))
        node_copy.setdefault("importance", "main")
        if not isinstance(node_copy.get("characters_involved"), list):
            node_copy["characters_involved"] = []
        node_copy["foreshadowing"] = bool(node_copy.get("foreshadowing", False))
        local_alias = f"new_node_{index}"
        if coverage_contract is not None:
            node_copy["canon_evidence"] = deepcopy(
                coverage_evidence_by_alias.get(local_alias, [])
            )
        local_id_map[local_alias] = current_max_id
        created_nodes.append(node_copy)

    if not created_nodes:
        raise ValueError("Reverse canon extraction produced no scene-anchored plot node.")

    all_nodes_by_id = {
        int(node["id"]): node
        for node in work_plots.get("plots_node", []) + created_nodes
        if isinstance(node, dict) and node.get("id") is not None
    }

    def resolve_plot_id(value) -> int | None:
        if isinstance(value, str) and value in local_id_map:
            return local_id_map[value]
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            if is_real_plot_ref(value):
                return parse_plot_ref(value)
            if value.strip().isdigit():
                return int(value.strip())
        return None

    created_relationships: list[list[str]] = []
    relationship_keys: set[tuple[str, str, str]] = set()

    def append_relationship(source_id: int, target_id: int, relation_type: str) -> None:
        if source_id == target_id or source_id not in all_nodes_by_id or target_id not in all_nodes_by_id:
            return
        if relation_type == "e_res" and not all_nodes_by_id[target_id].get("foreshadowing"):
            return
        key = (str(source_id), str(target_id), relation_type)
        if key not in relationship_keys:
            relationship_keys.add(key)
            created_relationships.append(list(key))

    for rel in patch.get("plots_relationship", []):
        if not isinstance(rel, list) or len(rel) < 3:
            continue
        source_raw, target_raw, rel_type = rel[0], rel[1], rel[2]
        canonical_type = normalize_edge_type(rel_type)
        if canonical_type is None or canonical_type == "e_temp":
            continue
        source_id = resolve_plot_id(source_raw)
        target_id = resolve_plot_id(target_raw)
        if source_id is None or target_id is None:
            continue
        append_relationship(source_id, target_id, canonical_type)

    previous_id = work_plots.get("_meta", {}).get("last_canon_plot_id")
    try:
        previous_id = int(previous_id) if previous_id is not None else None
    except (TypeError, ValueError):
        previous_id = None
    for node in created_nodes:
        current_id = int(node["id"])
        if previous_id is not None:
            append_relationship(previous_id, current_id, "e_temp")
        previous_id = current_id

    work_plots.setdefault("plots_node", []).extend(created_nodes)
    existing_relationship_keys = {
        (str(rel[0]), str(rel[1]), str(rel[2]))
        for rel in work_plots.get("plots_relationship", [])
        if isinstance(rel, list) and len(rel) >= 3
    }
    relationships_to_add = [
        rel for rel in created_relationships if tuple(rel) not in existing_relationship_keys
    ]
    work_plots.setdefault("plots_relationship", []).extend(relationships_to_add)
    work_plots.setdefault("_meta", {})["last_canon_plot_id"] = previous_id
    work_plots.setdefault("canon_plot_layer", []).append({
        "chapter_no": chapter_payload.get("chapter_no"),
        "summary": patch.get("summary", ""),
        "created_plot_ids": [make_plot_ref(node["id"]) for node in created_nodes],
        "relationships": relationships_to_add,
        "plan_coverage_contract": coverage_contract,
        "plan_coverage": deepcopy(coverage_items) if coverage_contract is not None else [],
    })
    if not set_plots_graph_data(work_premise_id, work_plots):
        raise OSError(f"Failed to persist reverse-canon patch for {work_premise_id}.")
    return {
        "created_nodes": created_nodes,
        "created_relationships": relationships_to_add,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _manager_call_succeeded(result) -> bool:
    if result is None or result is False:
        return False
    if isinstance(result, str):
        normalized = result.strip()
        if not normalized:
            return False
        return not (
            normalized.startswith("\u9519\u8bef")
            or normalized.casefold().startswith("error")
        )
    return bool(result)


def _replace_promoted_refs(value, reference_map: dict[str, str]):
    """Recursively replace exact temp refs while retaining seed `temp_id` provenance."""
    if isinstance(value, str):
        return reference_map.get(value, value)
    if isinstance(value, list):
        return [_replace_promoted_refs(item, reference_map) for item in value]
    if isinstance(value, dict):
        return {
            key: (item if key == "temp_id" else _replace_promoted_refs(item, reference_map))
            for key, item in value.items()
        }
    return value


def _replace_graph_promoted_refs(value, reference_map: dict[str, str]):
    """Replace exact structured refs in graph values and dictionary keys."""
    if isinstance(value, str):
        return reference_map.get(value, value)
    if isinstance(value, list):
        return [_replace_graph_promoted_refs(item, reference_map) for item in value]
    if isinstance(value, dict):
        promoted = {}
        for key, item in value.items():
            promoted_key = reference_map.get(key, key) if isinstance(key, str) else key
            promoted[promoted_key] = _replace_graph_promoted_refs(item, reference_map)
        return promoted
    return value


def _promote_event_references(
    event_payload: dict,
    character_ref_map: dict[str, str],
    environment_ref_map: dict[str, str],
) -> None:
    """Promote temp ids everywhere the next Event or Writer can consume them."""
    reference_map = {**character_ref_map, **environment_ref_map}

    # Replace chapter refs, expected deltas, R_active and other exact reference
    # values, but preserve the original temp_id field as trace provenance.
    promoted = _replace_promoted_refs(event_payload, reference_map)
    event_payload.clear()
    event_payload.update(promoted)

    seed_subgraph = event_payload.get("seed_subgraph", {})
    for entity_type, ref_map in (
        ("characters", character_ref_map),
        ("environments", environment_ref_map),
    ):
        section = seed_subgraph.get(entity_type, {})
        existing = section.setdefault("existing", [])
        for item in section.get("new", []):
            if not isinstance(item, dict):
                continue
            promoted_ref = ref_map.get(item.get("temp_id"))
            if promoted_ref:
                item["promoted_ref"] = promoted_ref
                if promoted_ref not in existing:
                    existing.append(promoted_ref)


def _restore_global_snapshots(premise_id: str, snapshots: dict[str, dict]) -> None:
    failures = []
    if not set_characters_graph_data(premise_id, snapshots["characters"]):
        failures.append("G_C")
    if not set_environments_graph_data(premise_id, snapshots["environments"]):
        failures.append("T_E")
    if not set_plots_graph_data(premise_id, snapshots["plots"]):
        failures.append("C_P")
    if failures:
        raise OSError(f"Failed to restore global snapshots: {', '.join(failures)}")


@_locked_by_premise
def sync_to_global_memory(premise_id: str, event_payload: dict, event_index: int) -> None:
    work_premise_id = make_work_premise_id(premise_id, event_index)
    work_characters = _normalize_character_graph(
        characters_graph_data(work_premise_id), f"working G_C ({work_premise_id})"
    )
    _, work_environments = _load_environment_graph(
        work_premise_id, f"working T_E ({work_premise_id})"
    )
    work_plots = _normalize_plot_graph(
        plots_graph_data(work_premise_id), f"working C_P ({work_premise_id})"
    )
    work_meta_path = _work_meta_path(work_premise_id)
    work_meta_snapshot = get_work_meta(work_premise_id, required=True)
    event_snapshot = deepcopy(event_payload)

    raw_global_characters = _require_graph_object(
        characters_graph_data(premise_id), "global G_C snapshot"
    )
    global_characters = _normalize_character_graph(
        raw_global_characters, "global G_C snapshot"
    )
    raw_global_environments, global_environments = _load_environment_graph(
        premise_id, "global T_E snapshot"
    )
    raw_global_plots = _require_graph_object(
        plots_graph_data(premise_id), "global C_P snapshot"
    )
    global_plots = _normalize_plot_graph(raw_global_plots, "global C_P snapshot")
    snapshots = {
        "characters": deepcopy(raw_global_characters),
        "environments": deepcopy(raw_global_environments),
        "plots": deepcopy(raw_global_plots),
    }

    _validate_environment_projection(
        work_environments,
        {
            "phys_root_id": global_environments.get("phys_root_id"),
            "conc_root_id": global_environments.get("conc_root_id"),
        },
    )

    from tools.character_graph_manager import CharacterGraphManager
    from tools.environment_graph_manager import EnvironmentTreeManager
    from tools.plot_graph_manager import PlotGraphManager

    try:
        global_ca = CharacterGraphManager(premise_id)
        global_ea = EnvironmentTreeManager(premise_id)
        global_pa = PlotGraphManager(premise_id)
        # The validated snapshots are the single transaction baseline. Manager
        # constructors may read the files for their own setup, but must not create
        # a second, racy view of global memory.
        global_ca.characters_graph = deepcopy(global_characters)
        global_ea.environments_graph = deepcopy(global_environments)
        global_pa.plots_graph = deepcopy(global_plots)

        character_id_map: dict[int, int] = {}
        for node in work_characters.get("characters_node", []):
            work_id = int(node["id"])
            global_index = global_ca.find_character_index_by_id(work_id)
            if global_index != -1:
                result = global_ca.update_character_info(work_id, deepcopy(node))
                if not _manager_call_succeeded(result):
                    raise ValueError(result)
                character_id_map[work_id] = work_id
                continue

            existing_id = global_ca.find_character_id_by_name(node["name"])
            if existing_id is not None:
                result = global_ca.update_character_info(existing_id, deepcopy(node))
                if not _manager_call_succeeded(result):
                    raise ValueError(result)
                character_id_map[work_id] = existing_id
                continue

            new_node = deepcopy(node)
            new_node["id"] = global_ca.get_max_character_id() + 1
            result = global_ca.add_characters([new_node])
            if not _manager_call_succeeded(result):
                raise ValueError(result)
            character_id_map[work_id] = int(new_node["id"])

        for rel in work_characters.get("characters_relationship", []):
            mapped_rel = deepcopy(rel)
            mapped_rel["source"] = character_id_map.get(int(rel["source"]), int(rel["source"]))
            mapped_rel["target"] = character_id_map.get(int(rel["target"]), int(rel["target"]))
            if global_ca.find_two_characters_relationship_by_id(mapped_rel["source"], mapped_rel["target"]):
                result = global_ca.update_relationship_by_edge(mapped_rel)
            else:
                result = global_ca.add_relationship(mapped_rel)
            if not _manager_call_succeeded(result):
                raise ValueError(result)

        environment_id_map: dict[str, str] = {}
        global_environment_nodes = global_ea.environments_graph.get("environments_node", [])
        global_environment_ids = {n["id"]: n for n in global_environment_nodes if n.get("id")}
        global_environment_names = {n["name"]: n for n in global_environment_nodes if n.get("name")}
        pending_new_environments: list[dict] = []

        for node in work_environments.get("environments_node", []):
            work_env_id = node.get("id")
            if work_env_id in global_environment_ids:
                result = global_ea.update_environment_info(work_env_id, deepcopy(node))
                if not _manager_call_succeeded(result):
                    raise ValueError(result)
                environment_id_map[work_env_id] = work_env_id
            elif node.get("name") in global_environment_names:
                real_id = global_environment_names[node["name"]]["id"]
                result = global_ea.update_environment_info(real_id, deepcopy(node))
                if not _manager_call_succeeded(result):
                    raise ValueError(result)
                environment_id_map[work_env_id] = real_id
            else:
                pending_new_environments.append(deepcopy(node))

        while pending_new_environments:
            remaining: list[dict] = []
            progressed = False
            for node in pending_new_environments:
                work_env_id = node.get("id")
                work_parent_id = node.get("parent_id")
                parent_id = environment_id_map.get(work_parent_id)
                if work_parent_id and parent_id is None:
                    remaining.append(node)
                    continue
                if node.get("branch") == CONC_BRANCH:
                    result = global_ea.add_conc_node(
                        name=node.get("name", ""),
                        parent_id=parent_id,
                        description=node.get("description", ""),
                        attributes=node.get("attributes", {}),
                    )
                else:
                    result = global_ea.add_phys_node(
                        name=node.get("name", ""),
                        parent_id=parent_id,
                        description=node.get("description", ""),
                        location=node.get("location", ""),
                        minutia=node.get("minutia", []),
                        atmosphere=node.get("atmosphere", ""),
                    )
                if not _manager_call_succeeded(result):
                    raise ValueError(result)
                environment_id_map[work_env_id] = str(result)
                progressed = True
            if not progressed:
                unresolved = [node.get("id") for node in remaining]
                raise ValueError(f"Cannot resolve new environment parent chain: {unresolved}")
            pending_new_environments = remaining

        global_ea.environments_graph, _ = ensure_environment_graph_ids(global_ea.environments_graph)
        _validate_environment_projection(global_ea.environments_graph)

        plot_id_map: dict[int, int] = {}
        for node in work_plots.get("plots_node", []):
            work_id = int(node["id"])
            if global_pa.find_plot_by_id(work_id):
                update_payload = {k: v for k, v in deepcopy(node).items() if k != "id"}
                if update_payload:
                    result = global_pa.update_plot_info(work_id, update_payload)
                    if not _manager_call_succeeded(result):
                        raise ValueError(result)
                plot_id_map[work_id] = work_id
                continue
            new_node = deepcopy(node)
            new_id = work_id
            if global_pa.find_plot_by_id(new_id):
                new_id = global_pa.get_max_plot_id() + 1
            new_node["id"] = new_id
            global_pa.plots_graph.setdefault("plots_node", []).append(new_node)
            plot_id_map[work_id] = new_id

        global_rel_keys = {
            tuple(str(item) for item in rel[:3])
            for rel in global_pa.plots_graph.get("plots_relationship", [])
            if isinstance(rel, list) and len(rel) >= 3
        }
        for rel in work_plots.get("plots_relationship", []):
            if not isinstance(rel, list) or len(rel) < 3:
                continue
            source = plot_id_map.get(int(rel[0]), int(rel[0]))
            target = plot_id_map.get(int(rel[1]), int(rel[1]))
            rel_tuple = (str(source), str(target), str(rel[2]))
            if rel_tuple not in global_rel_keys:
                global_pa.plots_graph.setdefault("plots_relationship", []).append(list(rel_tuple))
                global_rel_keys.add(rel_tuple)

        character_temp_to_work = work_characters.get("_meta", {}).get("temp_to_id", {})
        environment_temp_to_work = work_meta_snapshot.get("environment_temp_to_id", {})
        character_ref_map = {
            temp_ref: make_character_ref(character_id_map[int(work_id)])
            for temp_ref, work_id in character_temp_to_work.items()
            if int(work_id) in character_id_map
        }
        environment_ref_map = {
            temp_ref: environment_id_map[work_id]
            for temp_ref, work_id in environment_temp_to_work.items()
            if work_id in environment_id_map
        }
        if any(not is_real_character_ref(ref) for ref in character_ref_map.values()):
            raise ValueError(f"Invalid promoted character refs: {character_ref_map}")
        if any(not is_real_environment_ref(ref) for ref in environment_ref_map.values()):
            raise ValueError(f"Invalid promoted environment refs: {environment_ref_map}")

        graph_reference_map = {**character_ref_map, **environment_ref_map}
        global_ca.characters_graph = _replace_graph_promoted_refs(
            global_ca.characters_graph, graph_reference_map
        )
        global_ea.environments_graph = _replace_graph_promoted_refs(
            global_ea.environments_graph, graph_reference_map
        )
        global_pa.plots_graph = _replace_graph_promoted_refs(
            global_pa.plots_graph, graph_reference_map
        )
        _promote_event_references(event_payload, character_ref_map, environment_ref_map)

        promoted_work_meta = deepcopy(work_meta_snapshot)
        promoted_work_meta["global_ref_map"] = {
            **character_ref_map,
            **environment_ref_map,
        }
        if not set_file(work_meta_path, promoted_work_meta):
            raise OSError(f"Failed to persist promoted reference map for {work_premise_id}.")

        if not set_characters_graph_data(premise_id, global_ca.characters_graph):
            raise OSError("Failed to commit global G_C.")
        if not set_environments_graph_data(premise_id, global_ea.environments_graph):
            raise OSError("Failed to commit global T_E.")
        if not set_plots_graph_data(premise_id, global_pa.plots_graph):
            raise OSError("Failed to commit global C_P.")
    except Exception as sync_error:
        event_payload.clear()
        event_payload.update(event_snapshot)
        rollback_failures: list[str] = []
        if not set_file(work_meta_path, work_meta_snapshot):
            rollback_failures.append("working_meta")
        try:
            _restore_global_snapshots(premise_id, snapshots)
        except Exception as restore_error:
            rollback_failures.append(str(restore_error))
        if rollback_failures:
            raise OSError(
                "Global sync failed and rollback was incomplete: "
                + "; ".join(rollback_failures)
            ) from sync_error
        raise


init_event_work_graph = extract_working_substructure
merge_event_work_graph = sync_to_global_memory
