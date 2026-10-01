
import json
import os
import re
import time
from copy import deepcopy


ENABLE_FORESHADOWING_INJECTION = os.getenv("ENABLE_FORESHADOWING", "1") == "1"
ENABLE_HIJ_INJECTION = os.getenv("ENABLE_HIJ", "1") == "1"


OUTPUT_LANG = os.getenv("OUTPUT_LANG", "en").lower()


from agents import plan_agent, writer_agent, plot_agent
from data import (
    data_path,
    get_file,
    set_file,
    characters_graph_data,
    raw_environments_graph_data,
    plots_graph_data,
    set_characters_graph_data,
    set_environments_graph_data,
    set_plots_graph_data,
)
from storyline import generate_storyline
from tools import event_schema, extract_json
from tools.agent_event_logger import AgentEventLogger, make_trace_run_id, get_trace_logger
from tools.event_schema import extract_active_reference_set
from tools.graph_refs import (
    is_real_character_ref,
    is_real_environment_ref,
    is_real_plot_ref,
    parse_character_ref,
    make_character_ref,
)
from tools.working_memory import (
    apply_plot_canon_patch,
    extract_working_substructure,
    sync_to_global_memory,
    validate_environment_projection,
)


def is_transient_api_error(exc: Exception) -> bool:
    error_text = str(exc)
    error_name = exc.__class__.__name__
    transient_signals = [
        "502", "503", "504", "Bad Gateway", "gateway",
        "timeout", "temporarily unavailable",
        "InternalServerError", "APIConnectionError", "RateLimitError",
    ]
    return any(s.lower() in (error_name + " " + error_text).lower() for s in transient_signals)


def call_with_retry(func, *args, retries: int = 3, delay_seconds: int = 5, retry_trace_logger=None, **kwargs):
    trace = get_trace_logger(retry_trace_logger)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if not is_transient_api_error(exc) or attempt == retries:
                trace.log(
                    "error",
                    agent="System",
                    content=f"{getattr(func, '__name__', str(func))} call failed",
                    payload={"attempt": attempt, "retries": retries},
                    error=exc,
                    status="error",
                )
                raise
            print(f">>> Transient API error on attempt {attempt}; retrying in {delay_seconds} seconds. Error: {exc}")
            trace.log(
                "retry",
                agent="System",
                content=f"{getattr(func, '__name__', str(func))} transient error; preparing to retry",
                payload={"attempt": attempt, "retries": retries, "delay_seconds": delay_seconds},
                error=exc,
                status="retry",
            )
            time.sleep(delay_seconds)
    raise last_error


def _read_graph_snapshot(
    graph_name: str,
    loader,
    premise_id,
    required_list_fields: tuple[str, ...],
    required_root_fields: tuple[str, ...] = (),
) -> dict:
    try:
        snapshot = loader(premise_id)
    except Exception as exc:
        raise RuntimeError(f"Cannot read the {graph_name} snapshot: {exc}") from exc
    if not isinstance(snapshot, dict):
        raise RuntimeError(
            f"Cannot read the {graph_name} snapshot: expected a JSON object, got {type(snapshot).__name__}."
        )
    for field_name in required_list_fields:
        if not isinstance(snapshot.get(field_name), list):
            raise RuntimeError(f"Cannot read the {graph_name} snapshot: {field_name} must be an array.")

    if required_root_fields:
        node_ids = {
            str(node.get("id"))
            for node in snapshot.get("environments_node", [])
            if isinstance(node, dict) and node.get("id") is not None
        }
        for field_name in required_root_fields:
            root_id = snapshot.get(field_name)
            if not isinstance(root_id, str) or not root_id.strip() or root_id not in node_ids:
                raise RuntimeError(
                    f"Cannot read the {graph_name} snapshot: {field_name} must reference a root node that exists in the graph."
                )
    return deepcopy(snapshot)


def _restore_graph_snapshot(graph_name: str, setter, premise_id, snapshot: dict) -> str | None:
    try:
        restored = setter(premise_id, deepcopy(snapshot))
    except Exception as exc:
        return f"{graph_name} setter raised {exc.__class__.__name__}: {exc}"
    if restored is not True:
        return f"{graph_name} setter returned {restored!r}"
    return None


def _capture_global_graph_snapshots(premise_id) -> dict[str, dict]:
    """Capture the three durable graphs before an Event commit/checkpoint transaction."""
    return {
        "G_C": _read_graph_snapshot(
            "G_C", characters_graph_data, premise_id,
            ("characters_node", "characters_relationship"),
        ),
        "T_E": _read_graph_snapshot(
            "T_E", raw_environments_graph_data, premise_id,
            ("environments_node",),
            ("phys_root_id", "conc_root_id"),
        ),
        "C_P": _read_graph_snapshot(
            "C_P", plots_graph_data, premise_id,
            ("plots_node", "plots_relationship"),
        ),
    }


def _restore_global_graph_snapshots(premise_id, snapshots: dict[str, dict]) -> list[str]:
    """Restore all durable graphs; return every restoration failure."""
    failures: list[str] = []
    for graph_name, setter in (
        ("C_P", set_plots_graph_data),
        ("T_E", set_environments_graph_data),
        ("G_C", set_characters_graph_data),
    ):
        snapshot = snapshots.get(graph_name)
        if not isinstance(snapshot, dict):
            failures.append(f"{graph_name} has no usable snapshot")
            continue
        failure = _restore_graph_snapshot(
            graph_name, setter, premise_id, snapshot
        )
        if failure:
            failures.append(failure)
    return failures


def graph_update_by_chapter(
    premise_id,
    chapter,
    chapter_num,
    characters_info,
    envs_info,
    chapter_payload,
    canon_patch,
    trace_logger=None,
    event_index=None,
):
    from tools.character_graph_manager import CharacterGraphManager
    from tools.environment_graph_manager import EnvironmentTreeManager

    character_snapshot = _read_graph_snapshot(
        "G_C", characters_graph_data, premise_id,
        ("characters_node", "characters_relationship"),
    )
    environment_snapshot = _read_graph_snapshot(
        "T_E", raw_environments_graph_data, premise_id,
        ("environments_node",),
        ("phys_root_id", "conc_root_id"),
    )
    plot_snapshot = _read_graph_snapshot(
        "C_P", plots_graph_data, premise_id,
        ("plots_node", "plots_relationship"),
    )

    trace = get_trace_logger(trace_logger)
    ca = CharacterGraphManager(premise_id)
    ea = EnvironmentTreeManager(premise_id)

    try:
        trace.log(
            "graph_update_start",
            agent="System",
            phase="graph_update",
            event_index=event_index,
            chapter_no=chapter_payload.get("chapter_no"),
            content="Starting updates to the character graph, T_E, and the C_P canon layer",
            payload={"chapter_num": chapter_num, "canon_patch": canon_patch},
        )
        characters_name = ca.update_graph_by_passage(chapter, characters_info)
        if ca.last_passage_update_ok is not True:
            detail = ca.last_passage_update_error or "Character graph update could not be confirmed by parsing"
            raise ValueError(f"G_C update failed: {detail}")
        print('G_C processing complete!')

        envs_name = ea.update_graph_by_passage(chapter)
        expected_env_refs = (chapter_payload.get("refs") or {}).get("environments") or []
        if expected_env_refs and not envs_name:
            raise ValueError(
                "T_E update failed: the chapter declares environment references, but passage extraction returned no physical scenes."
            )
        print('T_E processing complete!')

        apply_plot_canon_patch(
            work_premise_id=premise_id,
            chapter_payload=chapter_payload,
            patch=canon_patch,
        )
        print('C_P canon patch processing complete!')
        trace.log(
            "graph_update_done",
            agent="System",
            phase="graph_update",
            event_index=event_index,
            chapter_no=chapter_payload.get("chapter_no"),
            content="All three graph updates completed",
            payload={
                "chapter_num": chapter_num,
                "characters_name": locals().get("characters_name"),
                "envs_name": locals().get("envs_name"),
            },
            status="ok",
        )
    except Exception as exc:
        trace.log(
            "error",
            agent="System",
            phase="graph_update",
            event_index=event_index,
            chapter_no=chapter_payload.get("chapter_no"),
            content="Graph update failed; restoring all three graph snapshots",
            error=exc,
            status="error",
        )
        rollback_failures = []
        for graph_name, setter, snapshot in (
            ("C_P", set_plots_graph_data, plot_snapshot),
            ("T_E", set_environments_graph_data, environment_snapshot),
            ("G_C", set_characters_graph_data, character_snapshot),
        ):
            failure = _restore_graph_snapshot(graph_name, setter, premise_id, snapshot)
            if failure:
                rollback_failures.append(failure)
        if rollback_failures:
            failure_message = "; ".join(rollback_failures)
            print(f"The three graph snapshots were not fully restored: {failure_message}")
            trace.log(
                "rollback_error",
                agent="System",
                phase="graph_update",
                event_index=event_index,
                chapter_no=chapter_payload.get("chapter_no"),
                content="The three graph snapshots were not fully restored",
                payload={"rollback_failures": rollback_failures},
                error=exc,
                status="error",
            )
            raise RuntimeError(
                f"Graph update failed ({exc}), and the three graph snapshots were not fully restored: {failure_message}"
            ) from exc
        raise


def collect_inherited_active_refs(premise_id: str, previous_event_payload: dict | None) -> dict:
    g_c = _read_graph_snapshot(
        "G_C continuity",
        characters_graph_data,
        premise_id,
        ("characters_node", "characters_relationship"),
    )
    global_character_refs: set[str] = set()
    protagonists: list[str] = []
    for node in g_c.get("characters_node", []):
        cid = node.get("id")
        if cid is None:
            continue
        ref = make_character_ref(int(cid))
        global_character_refs.add(ref)
        importance = str(node.get("importance", "")).strip().lower()
        if importance in {"protagonist", "\u4e3b\u89d2"}:
            protagonists.append(ref)

    g_e = _read_graph_snapshot(
        "T_E continuity",
        raw_environments_graph_data,
        premise_id,
        ("environments_node",),
        ("phys_root_id", "conc_root_id"),
    )
    validate_environment_projection(g_e)
    global_environment_refs = {
        str(node.get("id"))
        for node in g_e.get("environments_node", [])
        if node.get("id")
    }

    previous_chapters = (
        previous_event_payload.get("chapters", [])
        if isinstance(previous_event_payload, dict)
        else []
    )
    last_refs = {}
    if previous_chapters and isinstance(previous_chapters[-1], dict):
        last_refs = previous_chapters[-1].get("refs", {}) or {}

    inherited_chars = list(protagonists)
    inherited_chars.extend(
        ref
        for ref in last_refs.get("characters", []) or []
        if is_real_character_ref(ref) and ref in global_character_refs
    )
    inherited_envs = [
        ref
        for ref in last_refs.get("environments", []) or []
        if is_real_environment_ref(ref) and ref in global_environment_refs
    ]

    g_p = _read_graph_snapshot(
        "C_P continuity",
        plots_graph_data,
        premise_id,
        ("plots_node", "plots_relationship"),
    )
    foreshadow_refs = [
        f"p_{int(node['id'])}"
        for node in sorted(
            g_p.get("plots_node", []),
            key=lambda item: (
                int(item.get("chapter", 0) or 0) if isinstance(item, dict) else 0,
                int(item.get("id", 0) or 0) if isinstance(item, dict) else 0,
            ),
        )
        if node.get("id") is not None and node.get("foreshadowing") is True
    ]
    resolved_refs = {
        f"p_{int(rel[1])}"
        for rel in g_p.get("plots_relationship", [])
        if isinstance(rel, list) and len(rel) >= 3 and rel[2] == "e_res"
    }
    inherited_plots = [
        ref for ref in last_refs.get("plots", []) or [] if is_real_plot_ref(ref)
    ]
    inherited_plots.extend(ref for ref in foreshadow_refs if ref not in resolved_refs)

    def bounded_unique(values, limit):
        return list(dict.fromkeys(values))[:limit]

    return {
        "characters": bounded_unique(inherited_chars, 6),
        "environments": bounded_unique(inherited_envs, 4),
        "plots": bounded_unique(inherited_plots, 8),
    }


import re as _re
_UNREGISTERED_REF_RE = _re.compile(
    r"Chapter (\d+) refs\.(\w+) contains an unregistered reference: (\S+)"
)


def _strip_unregistered_refs(event_payload: dict, errors: list[str]) -> tuple[list[str], list[tuple]]:
    chapters = event_payload.get("chapters", []) or []
    residual: list[str] = []
    stripped: list[tuple] = []
    for err in errors:
        m = _UNREGISTERED_REF_RE.match(err)
        if not m:
            residual.append(err)
            continue
        ch_idx = int(m.group(1)) - 1
        ref_type = m.group(2)
        ref = m.group(3)
        if 0 <= ch_idx < len(chapters):
            ref_list = chapters[ch_idx].get("refs", {}).get(ref_type, [])
            if isinstance(ref_list, list) and ref in ref_list:
                ref_list.remove(ref)
                stripped.append((ch_idx + 1, ref_type, ref))
    return residual, stripped


def _backfill_chapter_refs_by_name_scan(event_payload: dict, premise_id: str) -> None:
    if not isinstance(event_payload, dict):
        return
    chapters = event_payload.get("chapters") or []
    if not chapters:
        return

    g_c = _read_graph_snapshot(
        "G_C ref backfill",
        characters_graph_data,
        premise_id,
        ("characters_node", "characters_relationship"),
    )
    name_to_ref: dict[str, str] = {}
    for node in g_c.get("characters_node", []):
        cid = node.get("id")
        if cid is None:
            continue
        ref = make_character_ref(int(cid))
        name = (node.get("name") or "").strip()
        if name:
            name_to_ref[name] = ref
        for alias in node.get("aliases", []) or []:
            alias = (alias or "").strip()
            if alias:
                name_to_ref[alias] = ref
    if not name_to_ref:
        return

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        text = chapter.get("pure_plot", "") or ""
        if not text:
            continue
        refs = chapter.setdefault("refs", {"characters": [], "environments": [], "plots": []})
        ch_chars = list(refs.get("characters") or [])
        hits: list[str] = []
        for name, ref in name_to_ref.items():
            if name in text and ref not in ch_chars and ref not in hits:
                hits.append(ref)
        if hits:
            refs["characters"] = ch_chars + hits
            print(f"💡 [Refs Backfill] Chapter {chapter.get('chapter_no')} added characters found in pure_plot: {hits}")


_REF_LITERAL_PATTERN = re.compile(r"\b(?:c_tmp_\d+|c_\d+|e_tmp_\d+|e_\d+|p_tmp_\d+|p_\d+)\b")


def _sanitize_ref_literals_in_chapter(chapter_payload: dict, premise_id: str, work_premise_id: str) -> None:
    if not isinstance(chapter_payload, dict):
        return

    name_map: dict[str, str] = {}
    work_characters: dict | None = None
    work_environments: dict | None = None
    for src_pid in (premise_id, work_premise_id):
        g_c = _read_graph_snapshot(
            f"G_C ref sanitizer ({src_pid})",
            characters_graph_data,
            src_pid,
            ("characters_node", "characters_relationship"),
        )
        for node in g_c.get("characters_node", []):
            cid = node.get("id")
            name = (node.get("name") or "").strip()
            if cid is not None and name:
                name_map[make_character_ref(int(cid))] = name
        g_t = _read_graph_snapshot(
            f"T_E ref sanitizer ({src_pid})",
            raw_environments_graph_data,
            src_pid,
            ("environments_node",),
            ("phys_root_id", "conc_root_id"),
        )
        validate_environment_projection(g_t)
        for node in g_t.get("environments_node", []):
            eid = node.get("id")
            name = (node.get("name") or "").strip()
            if eid and name:
                name_map[str(eid)] = name
        if src_pid == work_premise_id:
            work_characters = g_c
            work_environments = g_t

    if work_characters is None or work_environments is None:
        raise RuntimeError(f"Cannot read the working graph to sanitize references: {work_premise_id}")
    work_meta_chars = work_characters.get("_meta", {})
    char_temp_to_id = work_meta_chars.get("temp_to_id", {}) or {}
    for node in work_characters.get("characters_node", []):
        for tmp_id, mapped_int in char_temp_to_id.items():
            if int(mapped_int) == int(node.get("id", -1)):
                name = (node.get("name") or "").strip()
                if name:
                    name_map[tmp_id] = name
    work_meta_envs = work_environments.get("_meta", {})
    env_temp_to_id = work_meta_envs.get("temp_to_id", {}) or {}
    for node in work_environments.get("environments_node", []):
        for tmp_id, mapped_eid in env_temp_to_id.items():
            if str(mapped_eid) == str(node.get("id", "")):
                name = (node.get("name") or "").strip()
                if name:
                    name_map[tmp_id] = name

    stripped: list[str] = []

    def _repl(m: "re.Match[str]") -> str:
        token = m.group(0)
        if token in name_map:
            return name_map[token]
        stripped.append(token)
        return ""

    def _clean(value):
        if isinstance(value, str):
            new_val = _REF_LITERAL_PATTERN.sub(_repl, value)
            new_val = re.sub(r"\s{2,}", " ", new_val)
            return new_val
        if isinstance(value, list):
            return [_clean(x) for x in value]
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items()}
        return value

    for field in ("pure_plot", "expected_deltas", "chapter_struct"):
        if field in chapter_payload:
            chapter_payload[field] = _clean(chapter_payload[field])

    if stripped:
        print(f"🧹 [Ref Sanitize] Chapter {chapter_payload.get('chapter_no')} removed unregistered reference literals: {stripped}")


def parse_event_output(event_output: str) -> dict:
    raw_event = extract_json.extract_json(llm_output=event_output)
    event_payload = event_schema.normalize_event_payload(raw_event)
    if not event_payload:
        raise ValueError("EventPlanner output could not be parsed as Event JSON.")
    errors = event_schema.validate_event_payload(event_payload)
    if errors:
        residual_errors, stripped = _strip_unregistered_refs(event_payload, errors)
        for ch_no, ref_type, ref in stripped:
            print(f"⚠️  [parse_event_output] graceful fallback: removed unregistered {ref} from Chapter {ch_no} refs.{ref_type}")
        if residual_errors:
            raise ValueError(f"Event structure validation failed (serious errors remain after fallback): {residual_errors}")
        post_strip_errors = event_schema.validate_event_payload(event_payload)
        if post_strip_errors:
            raise ValueError(f"Event structure validation failed after removing unregistered references: {post_strip_errors}")
    return event_payload


def save_generation_outputs(pid: str, story: str, events: list[dict]) -> None:
    story_dir = os.path.join(data_path(), str(pid))
    os.makedirs(story_dir, exist_ok=True)

    story_path = os.path.join(story_dir, "generated_story.txt")
    events_path = os.path.join(story_dir, "planned_events.json")

    def write_text_atomic(path: str, content: str) -> None:
        temp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)
        except Exception:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            raise

    events_existed = os.path.exists(events_path)
    previous_events = get_file(events_path) if events_existed else None
    if events_existed and previous_events is False:
        raise OSError(f"Cannot snapshot the previous Event output: {events_path}")
    if not set_file(events_path, events):
        raise OSError(f"Failed to write Event output: {events_path}")
    try:
        write_text_atomic(story_path, story)
    except Exception as story_error:
        if events_existed:
            restored = set_file(events_path, previous_events)
        else:
            try:
                os.remove(events_path)
                restored = True
            except OSError:
                restored = False
        if not restored:
            raise OSError(
                f"Story output failed, and the Event output snapshot could not be restored: {events_path}"
            ) from story_error
        raise

    flat_story_path = os.path.join(data_path(), f"generated_story_{pid}.txt")
    try:
        write_text_atomic(flat_story_path, story)
        print(f">>> Story copy written to: {flat_story_path}")
    except Exception as e:
        print(f"⚠️  Failed to write a copy to the data/ root ({flat_story_path}): {e}")


def _read_existing_json_for_preflight(path: str, label: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Pre-run check could not read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Pre-run check found {label} is not a JSON object: {path}")
    return value


def _assert_fresh_premise_state(pid) -> None:
    if os.getenv("TOPO_ALLOW_DIRTY_RERUN", "0") == "1":
        return

    root = data_path()
    story_dir = os.path.join(root, str(pid))
    reasons: list[str] = []

    characters_path = os.path.join(story_dir, "characters.json")
    if os.path.exists(characters_path):
        graph = _read_existing_json_for_preflight(characters_path, "G_C")
        if graph.get("characters_node") or graph.get("characters_relationship"):
            reasons.append("G_C already contains characters or relationships")

    environments_path = os.path.join(story_dir, "environments.json")
    if os.path.exists(environments_path):
        graph = _read_existing_json_for_preflight(environments_path, "T_E")
        root_ids = {graph.get("phys_root_id"), graph.get("conc_root_id")} - {None, ""}
        nodes = graph.get("environments_node")
        if not isinstance(nodes, list):
            raise RuntimeError(
                f"Pre-run check found T_E.environments_node is not an array: {environments_path}"
            )
        non_root_nodes = [
            node
            for node in nodes
            if not isinstance(node, dict) or node.get("id") not in root_ids
        ]
        if non_root_nodes:
            reasons.append("T_E already contains non-root physical or conceptual nodes")

    plots_path = os.path.join(story_dir, "plots.json")
    if os.path.exists(plots_path):
        graph = _read_existing_json_for_preflight(plots_path, "C_P")
        if graph.get("plots_node") or graph.get("plots_relationship"):
            reasons.append("C_P already contains plot nodes or edges")

    for filename in ("generated_story.txt", "planned_events.json"):
        path = os.path.join(story_dir, filename)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            reasons.append(f"Existing output {filename}")

    flat_story_path = os.path.join(root, f"generated_story_{pid}.txt")
    if os.path.exists(flat_story_path) and os.path.getsize(flat_story_path) > 0:
        reasons.append(os.path.basename(flat_story_path))

    if os.path.isdir(root):
        work_prefix = f"{pid}__work__"
        work_dirs = [
            name
            for name in os.listdir(root)
            if name.startswith(work_prefix)
            and os.path.isdir(os.path.join(root, name))
        ]
        if work_dirs:
            reasons.append(f"Residual working-subgraph directories: {', '.join(sorted(work_dirs))}")

    if reasons:
        raise RuntimeError(
            f"premise={pid} is not a clean one-shot experiment state: {'; '.join(reasons)}."
            " Archive or move this premise's existing global, working, and output data before rerunning; "
            "if you intend to continue from the existing state, set TOPO_ALLOW_DIRTY_RERUN=1."
        )


def run_story_generation(premise_data: dict, max_events: int | None = None):
    for pid, ptext in premise_data.items():
        trace_run_id = make_trace_run_id(str(pid))
        trace_logger = AgentEventLogger(pid, trace_run_id)
        ablation_config = {
            "enable_foreshadowing_injection": ENABLE_FORESHADOWING_INJECTION,
            "enable_hij_injection": ENABLE_HIJ_INJECTION,
        }
        print(f"⚙️  [Ablation Config] {ablation_config}")
        trace_logger.log(
            "run_start",
            agent="System",
            content=f"Starting generation for premise={pid}",
            payload={"premise_id": pid, "max_events": max_events, "ablation": ablation_config},
            status="running",
        )

        try:
            _assert_fresh_premise_state(pid)
        except Exception as exc:
            trace_logger.log(
                "run_error",
                agent="System",
                phase="preflight",
                content=f"premise={pid} pre-run state check failed",
                payload={"premise_id": pid},
                error=exc,
                status="error",
            )
            raise

        try:
            storyline = generate_storyline(
                pid,
                ptext,
                trace_logger=trace_logger,
                trace_run_id=trace_run_id,
            )
        except Exception as exc:
            trace_logger.log(
                "run_error",
                agent="System",
                phase="global_planning",
                content=f"Global Planner failed; stopping premise={pid}",
                payload={"premise_id": pid},
                error=exc,
                status="error",
            )
            raise
        if not storyline:
            error = RuntimeError("Global Planner did not return a valid outline.")
            trace_logger.log(
                "run_error",
                agent="System",
                phase="global_planning",
                content=f"Outline is empty; stopping premise={pid}",
                error=error,
                status="error",
            )
            raise error

        event_history: list[dict] = []
        chapter_history: list[str] = []
        story_text: str = ""
        global_chapter_num: int = 0
        previous_event_payload: dict | None = None

        for event_index, one_abstract in enumerate(storyline, start=1):
            if max_events is not None and event_index > max_events:
                break

            print('Current Event abstract:', one_abstract)
            event_abstract = (
                f"This is the {one_abstract['stage']} act of the three-act novel outline. "
                f"The event outline for this act is: {one_abstract['event']}."
            )
            trace_logger.log(
                "event_planning_start",
                agent="System",
                phase="reason_inquire",
                event_index=event_index,
                content=event_abstract,
                payload={"storyline_item": one_abstract},
            )

            try:
                inherited_active_refs = collect_inherited_active_refs(pid, previous_event_payload)
                planned_event_output = call_with_retry(
                    plan_agent.plan_cep_agent_discuss_episode_plot,
                    pid,
                    retry_trace_logger=trace_logger,
                    one_event_abstract=event_abstract,
                    premise=ptext,
                    first_event=(event_index == 1),
                    preevent=previous_event_payload or "",
                    trace_logger=trace_logger,
                    event_index=event_index,
                    inherited_active_refs=inherited_active_refs,
                    enable_foreshadowing_injection=ENABLE_FORESHADOWING_INJECTION,
                )
                event_payload = parse_event_output(planned_event_output)
                if ENABLE_HIJ_INJECTION:
                    _backfill_chapter_refs_by_name_scan(event_payload, pid)
            except Exception as exc:
                trace_logger.log(
                    "run_error",
                    agent="System",
                    phase="event_planning",
                    event_index=event_index,
                    content=f"Event {event_index} planning failed; stopping premise={pid}",
                    payload={"premise_id": pid, "event_index": event_index},
                    error=exc,
                    status="error",
                )
                raise
            trace_logger.log(
                "event_planning_done",
                agent="System",
                phase="reason_finalize",
                event_index=event_index,
                content=event_payload.get("event_meta", {}).get("abstract", ""),
                payload=event_payload,
                status="ok",
            )

            try:
                work_premise_id = extract_working_substructure(pid, event_payload, event_index)
            except Exception as exc:
                trace_logger.log(
                    "run_error",
                    agent="System",
                    phase="work_graph_init",
                    event_index=event_index,
                    content=f"Event {event_index} M_sub initialization failed; stopping premise={pid}",
                    payload={"premise_id": pid, "event_index": event_index},
                    error=exc,
                    status="error",
                )
                raise
            trace_logger.log(
                "work_graph_init",
                agent="System",
                phase="work_graph",
                event_index=event_index,
                content=f"M_sub initialized:{work_premise_id}",
                payload={
                    "work_premise_id": work_premise_id,
                    "R_active": event_payload.get("R_active"),
                },
                status="ok",
            )

            for chapter_payload in event_payload.get("chapters", []):
                global_chapter_num += 1
                previous_chapter_text = chapter_history[-1] if chapter_history else ""

                _sanitize_ref_literals_in_chapter(chapter_payload, pid, work_premise_id)

                try:
                    chapter_info = call_with_retry(
                        writer_agent.write_chapter_txt,
                        retry_trace_logger=trace_logger,
                        premise_id=work_premise_id,
                        premise=ptext,
                        event_payload=event_payload,
                        chapter_payload=chapter_payload,
                        pre_chapter_text=previous_chapter_text,
                        trace_logger=trace_logger,
                        event_index=event_index,
                        enable_foreshadowing_injection=ENABLE_FORESHADOWING_INJECTION,
                        enable_hij_injection=ENABLE_HIJ_INJECTION,
                        output_lang=OUTPUT_LANG,
                    )
                except Exception as exc:
                    trace_logger.log(
                        "run_error",
                        agent="System",
                        phase="chapter_write",
                        event_index=event_index,
                        chapter_no=chapter_payload.get("chapter_no"),
                        content=(
                            f"Event {event_index} Chapter {chapter_payload.get('chapter_no')} "
                            f"writing failed; stopping premise={pid}"
                        ),
                        payload={"premise_id": pid, "event_index": event_index},
                        error=exc,
                        status="error",
                    )
                    raise
                chapter_text = f"\n# Chapter {global_chapter_num}\n" + chapter_info["chapter_text"]
                print('>>> Chapter body:\n', chapter_text)
                story_text += chapter_text
                chapter_history.append(chapter_text)

                try:
                    canon_patch = call_with_retry(
                        plot_agent.build_canon_plot_patch,
                        retry_trace_logger=trace_logger,
                        chapter_text=chapter_text,
                        chapter_payload=chapter_payload,
                        trace_logger=trace_logger,
                        event_index=event_index,
                    )

                    graph_update_by_chapter(
                        premise_id=work_premise_id,
                        chapter=chapter_text,
                        chapter_num=global_chapter_num,
                        characters_info=chapter_info["characters_info"],
                        envs_info=chapter_info["envs_info"],
                        chapter_payload=chapter_payload,
                        canon_patch=canon_patch,
                        trace_logger=trace_logger,
                        event_index=event_index,
                    )
                except Exception as exc:
                    trace_logger.log(
                        "run_error",
                        agent="System",
                        phase="chapter_hot_update",
                        event_index=event_index,
                        chapter_no=chapter_payload.get("chapter_no"),
                        content=(
                            f"Event {event_index} Chapter {chapter_payload.get('chapter_no')} "
                            f"reverse-canon/hot-update failed; stopping premise={pid}"
                        ),
                        payload={"premise_id": pid, "event_index": event_index},
                        error=exc,
                        status="error",
                    )
                    raise

            # Sync + output checkpoint is one durable Event transaction.  The
            # three-graph merge is internally atomic, but a later checkpoint
            # failure must also restore the pre-Event global snapshots; otherwise
            # dirty global memory would claim an Event that has no story/output.
            try:
                global_snapshots = _capture_global_graph_snapshots(pid)
            except Exception as exc:
                trace_logger.log(
                    "run_error",
                    agent="System",
                    phase="sync_snapshot",
                    event_index=event_index,
                    content=f"Event {event_index} pre-commit global snapshot failed; stopping premise={pid}",
                    payload={"premise_id": pid, "event_index": event_index},
                    error=exc,
                    status="error",
                )
                raise

            # Sync: M_sub -> M_global
            try:
                sync_to_global_memory(pid, event_payload, event_index)
            except Exception as exc:
                trace_logger.log(
                    "run_error",
                    agent="System",
                    phase="sync_to_global",
                    event_index=event_index,
                    content=f"Event {event_index} global commit failed; stopping premise={pid}",
                    payload={"premise_id": pid, "event_index": event_index},
                    error=exc,
                    status="error",
                )
                raise

            checkpoint_events = [*event_history, event_payload]
            try:
                save_generation_outputs(pid, story_text, checkpoint_events)
            except Exception as exc:
                rollback_failures = _restore_global_graph_snapshots(
                    pid, global_snapshots
                )
                if rollback_failures:
                    trace_logger.log(
                        "rollback_error",
                        agent="System",
                        phase="output_checkpoint",
                        event_index=event_index,
                        content="Output checkpoint failed; could not fully restore the three global graphs to their pre-Event state",
                        payload={
                            "premise_id": pid,
                            "event_index": event_index,
                            "rollback_failures": rollback_failures,
                        },
                        error=exc,
                        status="error",
                    )
                    raise RuntimeError(
                        f"Output checkpoint failed ({exc}), and rollback of the three global graphs was incomplete: "
                        + "; ".join(rollback_failures)
                    ) from exc
                trace_logger.log(
                    "work_graph_merge_rollback",
                    agent="System",
                    phase="output_checkpoint",
                    event_index=event_index,
                    content="Output checkpoint failed; the three global graphs have been restored to their pre-commit snapshots",
                    payload={"premise_id": pid, "event_index": event_index},
                    error=exc,
                    status="ok",
                )
                trace_logger.log(
                    "run_error",
                    agent="System",
                    phase="output_checkpoint",
                    event_index=event_index,
                    content=f"Event {event_index} output checkpoint failed; stopping premise={pid}",
                    payload={"premise_id": pid, "event_index": event_index},
                    error=exc,
                    status="error",
                )
                raise

            event_history.append(event_payload)
            previous_event_payload = event_payload
            trace_logger.log(
                "work_graph_merge",
                agent="System",
                phase="sync_to_global",
                event_index=event_index,
                content="Event M_sub and story output committed as a single checkpoint",
                payload={
                    "event_id": event_payload.get("event_meta", {}).get("event_id"),
                    "chapter_count": len(event_payload.get("chapters", [])),
                    "checkpoint_event_count": len(event_history),
                },
                status="ok",
            )

        try:
            save_generation_outputs(pid, story_text, event_history)
        except Exception as exc:
            trace_logger.log(
                "run_error",
                agent="System",
                phase="output_final",
                content=f"final output save failed; stopping premise={pid}",
                payload={"premise_id": pid, "completed_events": len(event_history)},
                error=exc,
                status="error",
            )
            raise
        trace_logger.log(
            "run_done",
            agent="System",
            content=f"Generation complete for premise={pid}",
            payload={
                "completed_events": len(event_history),
                "global_chapter_num": global_chapter_num,
            },
            status="completed",
        )
        print('>>> Complete story:\n', story_text)

        try:
            from tools.trace_digest import generate_trace_digest
            generate_trace_digest(pid, trace_run_id)
        except Exception as _digest_exc:
            print(f">>> Automatic trace_digest call failed (rerun manually with python -m tools.trace_digest {pid} {trace_run_id}): {_digest_exc}")


if __name__ == '__main__':
    start_time = time.time()

    with open(_wp_path, "r", encoding="utf-8") as _f:
        _wp_all = json.load(_f)
    premise_data = dict(list(_wp_all.items())[1:5])

    run_story_generation(premise_data, max_events=3)

    end_time = time.time()
    run_time = end_time - start_time
    minutes = int(run_time // 60)
    seconds = run_time % 60

    run_time_avg = run_time/len(premise_data)
    minutes_avg = int(run_time_avg // 60)
    seconds_avg = run_time_avg % 60
    if minutes == 0:
        print(f"Run time: {seconds:.4f} seconds")
    else:
        print(f"Average time per premise: {minutes_avg} minutes {seconds_avg:.4f} seconds")
        print(f"Total run time: {minutes} minutes {seconds:.4f} seconds")
