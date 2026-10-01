


import json
import re
import uuid
from copy import deepcopy
from functools import partial
from typing import Annotated, Any, List, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents import build_langchain_model_client
from agents.character_agent import (
    build_character_agent,
    propose_bootstrap_character,
    propose_character_creation,
)
from agents.environment_agent import (
    build_environment_agent,
    propose_bootstrap_environment,
    propose_environment_creation,
)
from agents.plot_agent import build_plot_agent
from tools import event_schema, extract_json
from tools.character_graph_manager import CharacterGraphManager
from tools.environment_graph_manager import EnvironmentTreeManager
from tools.environment_ids import PHYS_BRANCH
from tools.graph_refs import (
    CHARACTER_TEMP_PREFIX,
    ENVIRONMENT_TEMP_PREFIX,
    is_environment_ref,
    is_real_character_ref,
    is_real_environment_ref,
    is_real_plot_ref,
    is_temp_environment_ref,
    make_character_ref,
    make_temp_ref,
)
from tools.agent_event_logger import get_trace_logger


INFO_REQUEST_TAG = "InfoRequest"
EVENT_BLUEPRINT_TAG = "EventBlueprint"


PHASE_ORDER = [
    "reason_inquire",
    "reason_blueprint",
    "reason_finalize",
]


MAX_EVENT_ACTIVE_PLOT_REFS = 8
MAX_MANAGER_QUESTION_CHARS = 900
MAX_MANAGER_REPLY_CHARS = 4000

_REASON_MANAGER_REQUIREMENTS = (
    ("ca_replies_received", "CharacterManager", "discuss_with_ca"),
    ("ea_replies_received", "EnvironmentManager", "discuss_with_ea"),
    ("pa_replies_received", "PlotManager", "discuss_with_pa"),
)


def _ordered_unique_strings(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values or []:
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _build_seed_context(anchor_result: dict, inherited: dict | None) -> tuple[dict, list[str]]:
    anchor = anchor_result if isinstance(anchor_result, dict) else {}
    inherited = inherited if isinstance(inherited, dict) else {}
    protagonist_refs = _ordered_unique_strings(anchor.get("characters", []))
    seed_context = {
        "characters": _ordered_unique_strings(
            protagonist_refs + list(inherited.get("characters", []) or [])
        ),
        "environments": _ordered_unique_strings(
            list(anchor.get("environments", []) or [])
            + list(inherited.get("environments", []) or [])
        ),
        "plots": _ordered_unique_strings(inherited.get("plots", []) or []),
    }
    return seed_context, protagonist_refs


def _missing_reason_manager_replies(state: dict) -> list[tuple[str, str]]:
    return [
        (manager_name, tool_name)
        for counter_name, manager_name, tool_name in _REASON_MANAGER_REQUIREMENTS
        if int(state.get(counter_name, 0) or 0) <= 0
    ]


def _reason_inquire_stall_action(
    state: dict,
    step_count: int,
    soft_nudge_at: int,
    hard_fail_at: int,
) -> tuple[str, list[tuple[str, str]]]:
    missing = _missing_reason_manager_replies(state)
    if not missing:
        return "ready", missing
    if step_count >= hard_fail_at:
        return "fail", missing
    if step_count >= soft_nudge_at and not state.get("phase_nudge_sent", False):
        return "nudge", missing
    return "wait", missing


def _reason_inquire_nudge_message(missing: list[tuple[str, str]]) -> str:
    managers = ", ".join(manager for manager, _ in missing)
    tools = ", ".join(tool_name for _, tool_name in missing)
    return (
        "[System Dispatch] reason_inquire still lacks grounded replies from: "
        f"{managers}. Directly call the missing tool(s) now: {tools}. "
        "Give each tool one focused consistency question. Do not output prose, JSON, "
        "a tagged artifact, an EventBlueprint, or a final Event in this turn."
    )


def _manager_reply_is_grounded(
    prior_messages: list,
    result_messages: list,
    answer_content: Any,
) -> bool:
    if not isinstance(answer_content, str) or not answer_content.strip():
        return False
    if len(answer_content) > MAX_MANAGER_REPLY_CHARS:
        return False
    if not isinstance(prior_messages, list) or not isinstance(result_messages, list):
        return False
    new_messages = (
        result_messages[len(prior_messages):]
        if len(result_messages) >= len(prior_messages)
        else result_messages
    )
    return any(isinstance(message, ToolMessage) for message in new_messages)


def _manager_question_error(question: Any) -> str | None:
    if not isinstance(question, str) or not question.strip():
        return "Manager question must be a non-empty string."
    if len(question) > MAX_MANAGER_QUESTION_CHARS:
        return (
            f"Manager question is too broad ({len(question)} chars; max "
            f"{MAX_MANAGER_QUESTION_CHARS}). Ask one decision-oriented retrieval question."
        )
    return None


def _select_bounded_plot_inheritance(
    recent_plot_refs,
    inherited_plot_refs,
    unresolved_plot_refs,
    limit: int = MAX_EVENT_ACTIVE_PLOT_REFS,
) -> tuple[list[str], list[str]]:
    if limit < 0:
        raise ValueError("plot inheritance limit must be non-negative")
    ordered_candidates = _ordered_unique_strings(
        list(recent_plot_refs or [])
        + list(inherited_plot_refs or [])
        + list(unresolved_plot_refs or [])
    )
    bounded_active = [ref for ref in ordered_candidates if is_real_plot_ref(ref)][:limit]
    unresolved_set = {
        ref for ref in unresolved_plot_refs or [] if is_real_plot_ref(ref)
    }
    required_open = [ref for ref in bounded_active if ref in unresolved_set]
    return bounded_active, required_open


def _blueprint_requirements_are_well_formed(
    items,
    expected_prefix: str | None = None,
) -> bool:
    """Proposal requirements must have unique, explicit keys before phase advance."""
    if not isinstance(items, list):
        return False
    keys: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            return False
        key = str(item.get("requirement_key", "")).strip()
        if not key:
            return False
        if expected_prefix and not re.fullmatch(
            rf"{re.escape(expected_prefix)}\d+", key
        ):
            return False
        keys.append(key)
    return len(keys) == len(set(keys))


def _normalize_environment_proposal_requirement(requirement: dict) -> dict:
    """Validate and canonicalize the topology-bearing environment requirement."""
    if not isinstance(requirement, dict):
        raise ValueError("Environment requirement must be a JSON object.")
    normalized = deepcopy(requirement)
    _require_proposal_key(normalized, "env_req_")

    branch = str(normalized.get("branch") or "phys").strip().lower()
    if branch not in ("phys", "conc"):
        raise ValueError("Environment requirement branch must be 'phys' or 'conc'.")

    raw_parent = normalized.get("parent_hint")
    if raw_parent is None or (
        isinstance(raw_parent, str) and not raw_parent.strip()
    ):
        parent_hint = None
    elif not is_environment_ref(raw_parent):
        raise ValueError(
            "Environment requirement parent_hint must be null, e_K, or e_tmp_NN; "
            f"got {raw_parent!r}."
        )
    else:
        parent_hint = raw_parent.strip()

    temp_id = _temp_id_from_requirement(normalized, ENVIRONMENT_TEMP_PREFIX)
    if parent_hint == temp_id:
        raise ValueError(f"Environment proposal {temp_id} cannot be its own parent.")

    normalized["branch"] = branch
    normalized["parent_hint"] = parent_hint
    return normalized


def _validate_plot_inheritance_contract(
    event_payload: dict,
    required_open_threads,
    max_active_plots: int = MAX_EVENT_ACTIVE_PLOT_REFS,
) -> list[str]:
    if not isinstance(event_payload, dict):
        return ["Event payload must be an object for plot inheritance validation."]

    errors: list[str] = []
    active_plots = event_schema.extract_active_reference_set(event_payload).get("plots", [])
    if len(active_plots) > max_active_plots:
        errors.append(
            f"R_active.plots may contain at most {max_active_plots} unique p_K refs; "
            f"received {len(active_plots)}."
        )

    seed = event_payload.get("seed_subgraph", {})
    plots_seed = seed.get("plots", {}) if isinstance(seed, dict) else {}
    carried = set(
        plots_seed.get("unresolved_threads", [])
        if isinstance(plots_seed, dict)
        and isinstance(plots_seed.get("unresolved_threads", []), list)
        else []
    )
    payoff_refs: set[str] = set()
    for chapter in event_payload.get("chapters", []) or []:
        if not isinstance(chapter, dict):
            continue
        refs = chapter.get("refs", {})
        if isinstance(refs, dict) and isinstance(refs.get("plots", []), list):
            payoff_refs.update(refs.get("plots", []))

    for ref in _ordered_unique_strings(required_open_threads):
        if ref not in carried and ref not in payoff_refs:
            errors.append(
                f"Inherited open plot thread {ref} must either remain in "
                "seed_subgraph.plots.unresolved_threads or appear in a chapter refs.plots payoff."
            )
    return errors


def _next_phase(current: str) -> str | None:
    if current not in PHASE_ORDER:
        return None
    idx = PHASE_ORDER.index(current)
    return PHASE_ORDER[idx + 1] if idx + 1 < len(PHASE_ORDER) else None


_TAG_BLOCK_RES: dict[str, re.Pattern] = {}


_PLAN_TOOL_WHITELIST = frozenset({
    "discuss_with_ca", "discuss_with_ea", "discuss_with_pa",
    "request_character_proposal", "request_environment_proposal",
})


def _extract_spoofed_tool_calls(content: str) -> list[dict]:
    if not isinstance(content, str) or not content:
        return []

    parsed: Any = None
    if "multi_tool_use" in content:
        tail = content[content.find("multi_tool_use"):]
        try:
            parsed = extract_json.extract_json(tail)
        except Exception:
            parsed = None

    if not isinstance(parsed, dict):
        try:
            top = extract_json.extract_json(content)
            if isinstance(top, dict) and isinstance(top.get("tool_uses"), list):
                parsed = top
        except Exception:
            parsed = None

    if not isinstance(parsed, dict):
        return []
    uses = parsed.get("tool_uses") or parsed.get("tools")
    if not isinstance(uses, list) or not uses:
        return []

    tool_calls: list[dict] = []
    for use in uses:
        if not isinstance(use, dict):
            continue
        recipient = str(use.get("recipient_name") or use.get("name") or "").strip()
        if recipient.startswith("functions."):
            recipient = recipient[len("functions."):]
        if recipient not in _PLAN_TOOL_WHITELIST:
            continue
        params = use.get("parameters") or use.get("args") or {}
        if not isinstance(params, dict):
            continue
        tool_calls.append({
            "name": recipient,
            "id": f"spoof_{uuid.uuid4().hex[:10]}",
            "args": params,
        })
    return tool_calls


def _extract_tagged_json(text: str, tag: str) -> dict | None:
    if not isinstance(text, str) or not text:
        return None
    pattern = _TAG_BLOCK_RES.get(tag)
    if pattern is None:
        pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL)
        _TAG_BLOCK_RES[tag] = pattern
    match = pattern.search(text)
    if not match:
        return None
    raw = match.group(1).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = extract_json.extract_json(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _summarize_artifact(phase: str, artifact: dict) -> str:
    if not isinstance(artifact, dict) or not artifact:
        return "(no artifact recorded)"
    if phase == "reason_inquire":
        if artifact.get("force_advanced"):
            return "reason_inquire: phase timed out before any Manager response arrived (fallback path)"
        dispatched = artifact.get("dispatched", {})
        if dispatched:
            return (
                f"reason_inquire: Manager retrieval completed (dispatched ca={dispatched.get('ca', 0)} "
                f"ea={dispatched.get('ea', 0)} pa={dispatched.get('pa', 0)}; replies are in the message history)"
            )
        return "reason_inquire: Manager retrieval completed (replies are in the message history)"
    if phase == "reason_blueprint":
        blueprint = artifact.get("event_blueprint", {})
        proposals = artifact.get("proposals", {})
        return (
            f"EventBlueprint: chapter_count={blueprint.get('chapter_count')} "
            f"proposals=chars:{list(proposals.get('characters', {}).keys())} "
            f"envs:{list(proposals.get('environments', {}).keys())}"
        )
    return json.dumps(artifact, ensure_ascii=False)[:300]


class PlanAgent:
    def __init__(self, model_name=""):
        self.model_name = model_name
        self.history = []

    def chat(self, content):
        raise NotImplementedError("EventPlanner currently uses the LangGraph multi-agent mode.")


def _previous_event_summary(preevent) -> str:
    if isinstance(preevent, dict):
        normalized = event_schema.normalize_event_payload(preevent)
        if normalized:
            meta = normalized.get("event_meta", {})
            abstract = meta.get("abstract", "")
            exit_hook = meta.get("exit_hook", "")
            return f"{abstract}(exit_hook: {exit_hook})"
    if isinstance(preevent, str) and preevent.strip():
        try:
            parsed = extract_json.extract_json(preevent)
            normalized = event_schema.normalize_event_payload(parsed)
            if normalized:
                meta = normalized.get("event_meta", {})
                return f"{meta.get('abstract', '')}(exit_hook: {meta.get('exit_hook', '')})"
        except Exception:
            return ""
    return ""


def _previous_event_unresolved_threads(preevent, premise_id: str | None = None) -> list[str]:
    if not premise_id:
        return []
    try:
        from tools.plot_graph_manager import PlotGraphManager
        from tools.graph_refs import make_plot_ref
        plot_mgr = PlotGraphManager(premise_id)
        unresolved = plot_mgr.find_unresolved_foreshadowing()
        unresolved = sorted(
            (item for item in unresolved if isinstance(item, dict) and item.get("id") is not None),
            key=lambda item: (int(item.get("chapter", 0) or 0), int(item["id"])),
        )
        return [
            make_plot_ref(int(p["id"]))
            for p in unresolved
        ]
    except Exception as exc:
        raise RuntimeError(f"Failed to query unresolved threads in the global C_P: {exc}") from exc


def _previous_event_last_chapter(preevent) -> dict | None:
    normalized = None
    if isinstance(preevent, dict):
        normalized = event_schema.normalize_event_payload(preevent)
    elif isinstance(preevent, str) and preevent.strip():
        try:
            parsed = extract_json.extract_json(preevent)
            normalized = event_schema.normalize_event_payload(parsed)
        except Exception:
            return None
    if not isinstance(normalized, dict):
        return None
    chapters = normalized.get("chapters", []) or []
    return chapters[-1] if chapters and isinstance(chapters[-1], dict) else None


def _continuity_check_via_llm(parsed_event: dict, preevent) -> list[str]:
    if not isinstance(parsed_event, dict):
        return []
    chapters = parsed_event.get("chapters", []) or []
    items: list[dict] = []

    prev_last = _previous_event_last_chapter(preevent)
    if isinstance(prev_last, dict):
        items.append({
            "chapter_label": "Final chapter of previous Event",
            "pure_plot": prev_last.get("pure_plot", ""),
            "chapter_hook_out": prev_last.get("chapter_hook_out", ""),
            "refs": prev_last.get("refs", {}),
        })

    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        items.append({
            "chapter_label": f"Chapter {ch.get('chapter_no')}",
            "pure_plot": ch.get("pure_plot", ""),
            "chapter_hook_out": ch.get("chapter_hook_out", ""),
            "refs": ch.get("refs", {}),
        })

    if len(items) < 2:
        return []

    judge_prompt = (
        "You are a strict novel Continuity Critic.\n\n"
        "Task: Review the state continuity between adjacent chapters below "
        "and find any **hard contradictions that would seriously break the reader's immersion**.\n\n"
        "You must focus on (including but not limited to):\n"
        "1. [Location Jump] At the end of one chapter the character is at place A, at the start of the next chapter they are at place B, "
        "with no transitional explanation in either chapter (movement / being taken away / waking up already at... / fleeing to...).\n"
        "2. [Life/Death Contradiction] A character is explicitly killed, vanished, sacrificed, turned to ash, murdered, or permanently gone in one chapter, "
        "yet appears alive and normal in the next chapter — speaking or acting — with no reasonable explanation (hallucination / memory / resurrection mechanic, etc.).\n"
        "3. [Companion / Presence Discontinuity] People who evacuated / traveled / were escorted with the protagonist in the previous chapter "
        "suddenly disappear at the start of the next chapter, or are replaced by entirely different people, with no transition in between.\n"
        "4. [Item / Injury / Ability State Discontinuity] Key items carried in the previous chapter, severe injuries, locked / awakened ability states "
        "that suddenly disappear or reverse in the next chapter without explanation.\n"
        "5. [Broken Causality] Strong causal setups in the previous chapter (e.g., being wanted, factions declaring war, a key secret being exposed) "
        "to which the next chapter has zero response, with the plot turning to an unrelated direction.\n"
        "6. **Any other hard contradiction you judge would make the reader think 'wait, that's not right' — you must report it.**\n\n"
        "Notes:\n"
        "- Report only [hard contradictions]. Style differences, pacing, information-density differences, and unrecycled foreshadowing do NOT count.\n"
        "- If a jump has an explicit transitional explanation in the previous chapter's chapter_hook_out or in the next chapter's pure_plot, it is not a problem.\n"
        "- The boundary from a previous-Event final chapter to a new-Event first chapter must be checked under the same standard.\n\n"
        "Input (in chapter order):\n"
        f"{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        "Output format (strict JSON, no extra text):\n"
        "{\n"
        '  "has_continuity_issue": <true/false>,\n'
        '  "issues": [\n'
        "    {\n"
        '      "between": "<X → Y>",\n'
        '      "type": "<location_jump | character_death_revive | character_presence | item_state | causality | other>",\n'
        '      "description": "<one sentence describing the specific contradiction and the missing transition>"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    try:
        judge_client = build_langchain_model_client()
        resp = judge_client.invoke([HumanMessage(content=judge_prompt)])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        parsed = extract_json.extract_json(str(raw))
    except Exception:
        return []

    if not isinstance(parsed, dict) or not parsed.get("has_continuity_issue"):
        return []
    issues = parsed.get("issues", []) or []
    formatted: list[str] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        between = issue.get("between", "?")
        itype = issue.get("type", "other")
        desc = issue.get("description", "")
        formatted.append(f"[Continuity|{itype}] {between}: {desc}")
    return formatted


def _bootstrap_proposal_entry(
    ref_type: str,
    temp_id: str,
    requirement_key: str,
    proposal: dict,
) -> dict:
    entry = {
        "type": f"{ref_type}_create_proposal",
        "requirement_key": requirement_key,
        "temp_id": temp_id,
        "proposal": deepcopy(proposal),
    }
    if ref_type == "environment":
        # Bootstrap environments are authoritative root children.  Recording the
        # requirement in the same shape as normal env_req proposals lets the
        # final Event validator enforce one topology contract for both paths.
        entry["requirement"] = {
            "requirement_key": requirement_key,
            "branch": "phys",
            "parent_hint": None,
        }
    return entry


def ensure_protagonist_anchor(
    premise_id: str,
    premise: str,
    narrative_context: str = "",
    *,
    bootstrap_environment: bool = True,
    trace_logger=None,
    event_index=None,
) -> dict:
    trace = get_trace_logger(trace_logger)
    trace.log(
        "bootstrap_anchor_start",
        agent="System",
        phase="reason_inquire",
        event_index=event_index,
        content="Resolve the protagonist and first-scene anchors (read the global graph; propose new entities as temporary nodes only)",
    )

    char_manager = CharacterGraphManager(premise_id)
    env_manager = EnvironmentTreeManager(premise_id)
    proposals: dict[str, dict[str, dict]] = {"characters": {}, "environments": {}}

    protagonist = next(
        (
            node
            for node in char_manager.characters_graph.get("characters_node", [])
            if isinstance(node, dict)
            and str(node.get("importance", "")).strip().lower() in ("protagonist", "\u4e3b\u89d2")
        ),
        None,
    )
    if protagonist is not None:
        protagonist_ref = make_character_ref(protagonist["id"])
    else:
        trace.log(
            "bootstrap_proposal_start",
            agent="CharacterManager",
            phase="reason_blueprint",
            event_index=event_index,
            content="Global G_C has no protagonist; create an Event-local protagonist proposal",
        )
        proposal = propose_bootstrap_character(
            premise_id,
            premise,
            narrative_context=narrative_context,
        )
        protagonist_ref = make_temp_ref(CHARACTER_TEMP_PREFIX, 0)
        entry = _bootstrap_proposal_entry(
            "character",
            protagonist_ref,
            "bootstrap_protagonist",
            proposal,
        )
        proposals["characters"][protagonist_ref] = entry
        trace.log(
            "entity_proposal",
            agent="CharacterManager",
            to_agent="EventPlanner",
            phase="reason_blueprint",
            event_index=event_index,
            content=entry,
            payload=entry,
            status="ok",
        )

    environment_ref = None
    if bootstrap_environment:
        phys_root_id = env_manager.environments_graph.get("phys_root_id")
        phys_non_root = [
            node
            for node in env_manager.environments_graph.get("environments_node", [])
            if isinstance(node, dict)
            and node.get("branch") == PHYS_BRANCH
            and node.get("id") != phys_root_id
        ]
        if phys_non_root:
            environment_ref = phys_non_root[0].get("id")
        else:
            trace.log(
                "bootstrap_proposal_start",
                agent="EnvironmentManager",
                phase="reason_blueprint",
                event_index=event_index,
                content="Global T_phys has no first scene; create an Event-local environment proposal",
            )
            proposal = propose_bootstrap_environment(
                premise_id,
                premise,
                narrative_context=narrative_context,
            )
            environment_ref = make_temp_ref(ENVIRONMENT_TEMP_PREFIX, 0)
            entry = _bootstrap_proposal_entry(
                "environment",
                environment_ref,
                "bootstrap_environment",
                proposal,
            )
            proposals["environments"][environment_ref] = entry
            trace.log(
                "entity_proposal",
                agent="EnvironmentManager",
                to_agent="EventPlanner",
                phase="reason_blueprint",
                event_index=event_index,
                content=entry,
                payload=entry,
                status="ok",
            )

    result = {
        "characters": [protagonist_ref] if protagonist_ref else [],
        "environments": [environment_ref] if environment_ref else [],
        "plots": [],
        "_bootstrap_proposals": proposals,
    }
    trace.log(
        "bootstrap_anchor_done",
        agent="System",
        phase="reason_blueprint",
        event_index=event_index,
        content="Protagonist and first-scene anchors are ready",
        payload={
            "characters": result["characters"],
            "environments": result["environments"],
            "proposal_counts": {
                key: len(value) for key, value in proposals.items()
            },
        },
        status="ok",
    )
    return result


_bootstrap_core_entities = ensure_protagonist_anchor


def _parse_requirement_payload(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        parsed = extract_json.extract_json(payload)
        if isinstance(parsed, dict):
            return parsed
        try:
            return json.loads(payload)
        except Exception:
            return {"requirement": payload}
    return {}


def _require_proposal_key(requirement: dict, expected_prefix: str) -> str:
    key = str(requirement.get("requirement_key", "")).strip()
    if not key:
        raise ValueError(
            "requirement_key is required. Copy the exact requirement_key from the "
            "EventBlueprint into requirement_json before requesting a proposal."
        )
    if not re.fullmatch(rf"{re.escape(expected_prefix)}\d+", key):
        raise ValueError(
            f"requirement_key {key!r} must match {expected_prefix}<number>."
        )
    return key


def _temp_id_from_requirement(requirement: dict, prefix: str) -> str:
    key = str(requirement.get("requirement_key", "")).strip()
    match = re.search(r"(\d+)$", key)
    if match:
        return make_temp_ref(prefix, int(match.group(1)))
    return make_temp_ref(prefix, 1)


def _flatten_proposal_to_seed_node(
    prop_entry: dict, temp_id: str, first_ch: int, ref_type: str
) -> dict:
    inner = prop_entry.get("proposal") if isinstance(prop_entry, dict) else None
    recommended = inner.get("recommended") if isinstance(inner, dict) else {}
    if not isinstance(recommended, dict):
        recommended = {}
    profile = recommended.get("profile", {})
    if not isinstance(profile, dict):
        profile = {}
    node = {
        "temp_id": temp_id,
        "first_appearance_chapter": int(first_ch or 1),
        "name": recommended.get("name", "") or "",
        "profile": deepcopy(profile),
    }
    if ref_type == "characters":
        aliases = recommended.get("aliases")
        if isinstance(aliases, list) and aliases:
            node["aliases"] = [str(a).strip() for a in aliases if isinstance(a, (str, int)) and str(a).strip()]
    if ref_type == "environments":
        requirement = prop_entry.get("requirement") if isinstance(prop_entry, dict) else None
        if not isinstance(requirement, dict) and isinstance(inner, dict):
            requirement = inner.get("requirement")
        has_authoritative_parent = isinstance(requirement, dict) and "parent_hint" in requirement
        branch = (
            requirement.get("branch") if isinstance(requirement, dict) else None
        ) or (inner.get("branch") if isinstance(inner, dict) else None) or "phys"
        node["branch"] = branch
        parent_hint = (
            requirement.get("parent_hint")
            if has_authoritative_parent
            else recommended.get("parent_hint")
        )
        if isinstance(parent_hint, str):
            parent_hint = parent_hint.strip() or None
        # ``parent_id`` is the canonical seed/M_sub edge field.  Retaining
        # ``parent_hint`` makes the blueprint provenance explicit and allows the
        # validator to detect a model that emitted two contradictory fields.
        node["parent_id"] = parent_hint
        node["parent_hint"] = parent_hint
    return node


def _find_proposal_by_temp_id(type_proposals: dict, temp_id: str) -> dict | None:
    if not isinstance(type_proposals, dict):
        return None
    for prop_entry in type_proposals.values():
        if isinstance(prop_entry, dict) and prop_entry.get("temp_id") == temp_id:
            return prop_entry
    return None


def _backfill_seed_subgraph_from_proposals(
    event_payload: dict,
    phase_artifacts: dict,
    trace_logger=None,
    event_index=None,
) -> list[str]:
    if not isinstance(event_payload, dict):
        return []
    proposals_root = (
        phase_artifacts.get("reason_blueprint", {}).get("proposals", {})
        if isinstance(phase_artifacts, dict)
        else {}
    )
    if not isinstance(proposals_root, dict) or not proposals_root:
        return []

    chapter_first_seen: dict[str, dict[str, int]] = {
        "characters": {}, "environments": {},
    }
    for ch in event_payload.get("chapters", []) or []:
        if not isinstance(ch, dict):
            continue
        ch_no = ch.get("chapter_no") or 1
        refs = ch.get("refs", {}) or {}
        for ref_type in ("characters", "environments"):
            for ref in refs.get(ref_type, []) or []:
                if isinstance(ref, str) and "_tmp_" in ref:
                    chapter_first_seen[ref_type].setdefault(ref, ch_no)

    seed = event_payload.setdefault("seed_subgraph", {})
    backfilled: list[str] = []

    for ref_type in ("characters", "environments"):
        section = seed.setdefault(ref_type, {})
        if not isinstance(section.get("new"), list):
            section["new"] = []
        registered = {
            item["temp_id"]
            for item in section["new"]
            if isinstance(item, dict) and item.get("temp_id")
        }
        type_proposals = proposals_root.get(ref_type, {})
        if not isinstance(type_proposals, dict):
            continue
        for missing_ref, first_ch in chapter_first_seen[ref_type].items():
            if missing_ref in registered:
                continue
            prop_entry = _find_proposal_by_temp_id(type_proposals, missing_ref)
            if prop_entry is None:
                continue
            section["new"].append(
                _flatten_proposal_to_seed_node(prop_entry, missing_ref, first_ch, ref_type)
            )
            backfilled.append(f"{ref_type}:{missing_ref}")

    repaired: list[str] = []
    for ref_type in ("characters", "environments"):
        section = seed.get(ref_type, {}) or {}
        entries = section.get("new")
        if not isinstance(entries, list):
            continue
        type_proposals = proposals_root.get(ref_type, {})
        if not isinstance(type_proposals, dict):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            temp_id = entry.get("temp_id")
            if not temp_id:
                continue
            prop_entry = _find_proposal_by_temp_id(type_proposals, temp_id)
            if prop_entry is None:
                continue
            inner = prop_entry.get("proposal")
            recommended = inner.get("recommended") if isinstance(inner, dict) else None
            if not isinstance(recommended, dict):
                continue
            canonical_profile = recommended.get("profile", {})
            if not isinstance(entry.get("profile"), dict) and isinstance(canonical_profile, dict) and canonical_profile:
                entry["profile"] = deepcopy(canonical_profile)
                repaired.append(f"{ref_type}:{temp_id}:profile")
            if not entry.get("name"):
                name = recommended.get("name", "")
                if name:
                    entry["name"] = name
                    repaired.append(f"{ref_type}:{temp_id}:name")
            if ref_type == "characters":
                canonical_aliases = recommended.get("aliases")
                entry_aliases = entry.get("aliases")
                if isinstance(canonical_aliases, list) and canonical_aliases and not (
                    isinstance(entry_aliases, list) and entry_aliases
                ):
                    entry["aliases"] = [
                        str(a).strip()
                        for a in canonical_aliases
                        if isinstance(a, (str, int)) and str(a).strip()
                    ]
                    repaired.append(f"{ref_type}:{temp_id}:aliases")
            else:
                requirement = prop_entry.get("requirement")
                if not isinstance(requirement, dict) and isinstance(inner, dict):
                    requirement = inner.get("requirement")
                if isinstance(requirement, dict):
                    canonical_branch = str(requirement.get("branch") or "phys").strip().lower()
                    canonical_parent = requirement.get("parent_hint")
                    if isinstance(canonical_parent, str):
                        canonical_parent = canonical_parent.strip() or None
                    if entry.get("branch") != canonical_branch:
                        entry["branch"] = canonical_branch
                        repaired.append(f"{ref_type}:{temp_id}:branch")
                    if (
                        entry.get("parent_id") != canonical_parent
                        or entry.get("parent_hint") != canonical_parent
                    ):
                        entry["parent_id"] = canonical_parent
                        entry["parent_hint"] = canonical_parent
                        repaired.append(f"{ref_type}:{temp_id}:parent")

    if (backfilled or repaired) and trace_logger is not None:
        try:
            trace_logger.log(
                "seed_subgraph_backfill",
                agent="System",
                phase="reason_finalize",
                event_index=event_index,
                content=(
                    f"Backfilled {len(backfilled)} entries and repaired {len(repaired)} entries "
                    f"(preventing extract_working_substructure AttributeError or lost records)"
                ),
                payload={"backfilled": backfilled, "repaired": repaired},
                status="ok",
            )
        except Exception:
            pass
    return backfilled + repaired


def _environment_proposal_matches_requirement(
    needed_item: dict,
    proposal_entry: dict,
) -> bool:
    """Return whether a cached proposal preserves the exact blueprint topology."""
    if not isinstance(needed_item, dict) or not isinstance(proposal_entry, dict):
        return False
    try:
        expected = _normalize_environment_proposal_requirement(needed_item)
        actual_raw = proposal_entry.get("requirement")
        if not isinstance(actual_raw, dict):
            inner = proposal_entry.get("proposal")
            actual_raw = inner.get("requirement") if isinstance(inner, dict) else None
        actual = _normalize_environment_proposal_requirement(actual_raw)
    except (TypeError, ValueError):
        return False
    if not all(
        actual.get(field) == expected.get(field)
        for field in ("requirement_key", "branch", "parent_hint")
    ):
        return False

    inner = proposal_entry.get("proposal")
    if not isinstance(inner, dict):
        return False
    recommended = inner.get("recommended")
    if not isinstance(recommended, dict):
        return False
    proposal_parent = recommended.get("parent_hint")
    if isinstance(proposal_parent, str):
        proposal_parent = proposal_parent.strip() or None
    return (
        str(inner.get("branch") or "").strip().lower() == expected["branch"]
        and proposal_parent == expected["parent_hint"]
    )


def _proposal_satisfies_blueprint_requirement(
    needed_item: Any,
    proposals_dict: dict,
    prefix: str,
) -> bool:
    """Match one frozen blueprint requirement to its authoritative proposal."""
    if not isinstance(needed_item, dict) or not isinstance(proposals_dict, dict):
        return False
    req_key = str(needed_item.get("requirement_key", "")).strip()
    matched = proposals_dict.get(req_key) if req_key else None
    expected_temp = _temp_id_from_requirement(needed_item, prefix)
    if matched is None:
        matched = proposals_dict.get(expected_temp)
    if matched is None and req_key:
        for value in proposals_dict.values():
            if isinstance(value, dict) and value.get("requirement_key") == req_key:
                matched = value
                break
    if not isinstance(matched, dict):
        return False
    if prefix == ENVIRONMENT_TEMP_PREFIX:
        return _environment_proposal_matches_requirement(needed_item, matched)
    return matched.get("type") == "character_create_proposal"


def _blueprint_gate_diagnostics(
    blueprint: Any,
    proposals: Any,
    pending_tasks: Any = None,
) -> list[str]:
    """Explain every unmet reason_blueprint gate instead of returning a silent False."""
    if not isinstance(blueprint, dict) or not blueprint:
        return ["missing <EventBlueprint> JSON artifact"]
    diagnostics: list[str] = []
    needed_chars = blueprint.get("needed_new_characters", [])
    needed_envs = blueprint.get("needed_new_environments", [])
    if not _blueprint_requirements_are_well_formed(needed_chars, "char_req_"):
        diagnostics.append(
            "needed_new_characters must be an array with unique char_req_N requirement_key values"
        )
    if not _blueprint_requirements_are_well_formed(needed_envs, "env_req_"):
        diagnostics.append(
            "needed_new_environments must be an array with unique env_req_N requirement_key values"
        )
    proposal_groups = proposals if isinstance(proposals, dict) else {}
    char_proposals = proposal_groups.get("characters", {})
    env_proposals = proposal_groups.get("environments", {})
    if _blueprint_requirements_are_well_formed(needed_chars, "char_req_"):
        for item in needed_chars:
            if not _proposal_satisfies_blueprint_requirement(
                item, char_proposals, CHARACTER_TEMP_PREFIX
            ):
                diagnostics.append(
                    f"missing valid character proposal for {item['requirement_key']}"
                )
    if _blueprint_requirements_are_well_formed(needed_envs, "env_req_"):
        for item in needed_envs:
            if not _proposal_satisfies_blueprint_requirement(
                item, env_proposals, ENVIRONMENT_TEMP_PREFIX
            ):
                diagnostics.append(
                    f"missing topology-matching environment proposal for {item['requirement_key']}"
                )
    if pending_tasks:
        diagnostics.append(f"{len(pending_tasks)} pending Manager task(s) must return")
    return diagnostics


def _validate_environment_parent_contract(
    event_payload: dict,
    phase_artifacts: dict,
    global_environment_graph: dict | None = None,
) -> list[str]:
    """Validate proposal-backed environment topology before Event approval.

    The blueprint requirement is authoritative.  Every Event-local environment
    must retain its required branch/parent, temp parents must be registered and
    proposal-backed in the same branch, real parents must resolve in global T_E,
    and the temp subgraph must remain acyclic.
    """
    if not isinstance(event_payload, dict):
        return ["Environment parent contract requires an Event object."]

    errors: list[str] = []
    reason_blueprint = (
        phase_artifacts.get("reason_blueprint", {})
        if isinstance(phase_artifacts, dict)
        else {}
    )
    proposals_root = (
        reason_blueprint.get("proposals", {})
        if isinstance(reason_blueprint, dict)
        else {}
    )
    env_proposals = (
        proposals_root.get("environments", {})
        if isinstance(proposals_root, dict)
        else {}
    )
    if not isinstance(env_proposals, dict):
        env_proposals = {}
    proposals_by_temp = {
        entry.get("temp_id"): entry
        for entry in env_proposals.values()
        if isinstance(entry, dict) and is_temp_environment_ref(entry.get("temp_id"))
    }

    seed = event_payload.get("seed_subgraph", {})
    env_seed = seed.get("environments", {}) if isinstance(seed, dict) else {}
    new_items = env_seed.get("new", []) if isinstance(env_seed, dict) else []
    if not isinstance(new_items, list):
        return ["seed_subgraph.environments.new must be an array."]

    seed_by_temp: dict[str, dict] = {}
    parent_by_temp: dict[str, str | None] = {}
    branch_by_temp: dict[str, str] = {}

    def _canonical_parent(value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value.strip() if isinstance(value, str) else value

    for index, item in enumerate(new_items):
        path = f"seed_subgraph.environments.new[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{path} must be an object.")
            continue
        temp_id = item.get("temp_id")
        if not is_temp_environment_ref(temp_id):
            errors.append(f"{path}.temp_id must be a canonical e_tmp_NN ref: {temp_id!r}")
            continue
        if temp_id in seed_by_temp:
            errors.append(f"{path}.temp_id duplicates {temp_id}.")
            continue

        branch = str(item.get("branch") or "").strip().lower()
        if branch not in ("phys", "conc"):
            errors.append(f"{path}.branch must be 'phys' or 'conc'.")
        seed_by_temp[temp_id] = item
        branch_by_temp[temp_id] = branch

        has_parent_id = "parent_id" in item
        has_parent_hint = "parent_hint" in item
        parent_id = _canonical_parent(item.get("parent_id"))
        parent_hint = _canonical_parent(item.get("parent_hint"))
        if has_parent_id and has_parent_hint and parent_id != parent_hint:
            errors.append(
                f"{path} has contradictory parent_id={parent_id!r} and "
                f"parent_hint={parent_hint!r}."
            )
        parent = parent_id if has_parent_id else parent_hint
        parent_by_temp[temp_id] = parent
        if parent is not None and not is_environment_ref(parent):
            errors.append(
                f"{path} parent must be null, e_K, or e_tmp_NN; got {parent!r}."
            )

        proposal_entry = proposals_by_temp.get(temp_id)
        if proposal_entry is None:
            errors.append(f"{path} {temp_id} is not backed by an available environment proposal.")
            continue
        requirement = proposal_entry.get("requirement")
        inner = proposal_entry.get("proposal")
        if not isinstance(requirement, dict) and isinstance(inner, dict):
            requirement = inner.get("requirement")
        if not isinstance(requirement, dict):
            errors.append(f"Environment proposal {temp_id} is missing its authoritative requirement.")
            continue

        expected_branch = str(requirement.get("branch") or "").strip().lower()
        expected_parent = _canonical_parent(requirement.get("parent_hint"))
        if expected_branch not in ("phys", "conc"):
            errors.append(f"Environment proposal {temp_id} has invalid requirement branch.")
        if expected_parent is not None and not is_environment_ref(expected_parent):
            errors.append(
                f"Environment proposal {temp_id} has invalid requirement parent_hint "
                f"{expected_parent!r}."
            )
        if branch != expected_branch:
            errors.append(
                f"Environment {temp_id} branch {branch!r} does not match its "
                f"requirement branch {expected_branch!r}."
            )
        if parent != expected_parent:
            errors.append(
                f"Environment {temp_id} parent {parent!r} does not match its "
                f"requirement parent_hint {expected_parent!r}."
            )

        if isinstance(inner, dict):
            proposal_branch = str(inner.get("branch") or "").strip().lower()
            recommended = inner.get("recommended")
            proposal_parent = (
                _canonical_parent(recommended.get("parent_hint"))
                if isinstance(recommended, dict)
                else None
            )
            if proposal_branch != expected_branch:
                errors.append(
                    f"Environment proposal {temp_id} changed requirement branch "
                    f"{expected_branch!r} to {proposal_branch!r}."
                )
            if proposal_parent != expected_parent:
                errors.append(
                    f"Environment proposal {temp_id} changed requirement parent_hint "
                    f"{expected_parent!r} to {proposal_parent!r}."
                )
            nested_requirement = inner.get("requirement")
            if isinstance(nested_requirement, dict):
                nested_branch = str(
                    nested_requirement.get("branch") or ""
                ).strip().lower()
                nested_parent = _canonical_parent(
                    nested_requirement.get("parent_hint")
                )
                if (
                    nested_requirement.get("requirement_key")
                    != requirement.get("requirement_key")
                    or nested_branch != expected_branch
                    or nested_parent != expected_parent
                ):
                    errors.append(
                        f"Environment proposal {temp_id} does not preserve its "
                        "authoritative requirement object."
                    )

    global_nodes = {
        node.get("id"): node
        for node in (
            global_environment_graph.get("environments_node", [])
            if isinstance(global_environment_graph, dict)
            else []
        )
        if isinstance(node, dict) and node.get("id")
    }
    for temp_id, parent in parent_by_temp.items():
        branch = branch_by_temp.get(temp_id)
        if parent is None:
            continue
        if is_temp_environment_ref(parent):
            parent_item = seed_by_temp.get(parent)
            if parent_item is None:
                errors.append(
                    f"Environment {temp_id} requires unresolved temp parent {parent}."
                )
                continue
            if parent not in proposals_by_temp:
                errors.append(
                    f"Environment {temp_id} temp parent {parent} is not proposal-backed."
                )
            parent_branch = branch_by_temp.get(parent)
            if branch and parent_branch and parent_branch != branch:
                errors.append(
                    f"Environment {temp_id} ({branch}) cannot mount under {parent} "
                    f"({parent_branch}); branches differ."
                )
        elif is_real_environment_ref(parent):
            parent_node = global_nodes.get(parent)
            if parent_node is None:
                errors.append(
                    f"Environment {temp_id} requires unknown global parent {parent}."
                )
            elif parent_node.get("branch") != branch:
                errors.append(
                    f"Environment {temp_id} ({branch}) cannot mount under {parent} "
                    f"({parent_node.get('branch')}); branches differ."
                )

    # Each node has at most one parent, so a simple ancestor walk detects cycles.
    for temp_id in parent_by_temp:
        seen = {temp_id}
        current = parent_by_temp.get(temp_id)
        while is_temp_environment_ref(current):
            if current in seen:
                errors.append(f"Environment temp parent chain contains a cycle at {current}.")
                break
            seen.add(current)
            current = parent_by_temp.get(current)

    return errors


def _try_parse_event_candidate(content: Any) -> dict | None:
    try:
        parsed = extract_json.extract_json(str(content))
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    try:
        normalized = event_schema.normalize_event_payload(parsed)
    except Exception:
        return None
    if not isinstance(normalized, dict):
        return None
    return normalized


def plan_cep_agent_discuss_episode_plot(
    premise_id,
    one_event_abstract,
    premise,
    first_event=False,
    preevent="",
    trace_logger=None,
    event_index=None,
    inherited_active_refs: dict | None = None,
    enable_foreshadowing_injection: bool = True,
):

    trace = get_trace_logger(trace_logger)
    anchor_result = ensure_protagonist_anchor(
        premise_id,
        premise,
        narrative_context=one_event_abstract,
        bootstrap_environment=bool(first_event),
        trace_logger=trace,
        event_index=event_index,
    )
    bootstrap_proposals = deepcopy(anchor_result.get("_bootstrap_proposals", {}))
    ca = build_character_agent(premise_id)
    ea = build_environment_agent(premise_id)
    pa = build_plot_agent(premise_id)
    # Read-only topology snapshot used by the final Event validator to resolve
    # formal e_K parents and verify that parent/child branches agree.
    global_environment_contract_graph = deepcopy(
        EnvironmentTreeManager(premise_id).environments_graph
    )

    inherited = inherited_active_refs or {}
    bootstrap_result, true_protagonist_refs = _build_seed_context(anchor_result, inherited)
    bootstrap_character_seed_nodes = [
        _flatten_proposal_to_seed_node(entry, temp_id, 1, "characters")
        for temp_id, entry in (bootstrap_proposals.get("characters", {}) or {}).items()
        if isinstance(entry, dict)
    ]
    bootstrap_environment_seed_nodes = [
        _flatten_proposal_to_seed_node(entry, temp_id, 1, "environments")
        for temp_id, entry in (bootstrap_proposals.get("environments", {}) or {}).items()
        if isinstance(entry, dict)
    ]
    bootstrap_existing_characters = [
        ref for ref in bootstrap_result.get("characters", []) if is_real_character_ref(ref)
    ]
    bootstrap_existing_environments = [
        ref for ref in bootstrap_result.get("environments", []) if is_real_environment_ref(ref)
    ]
    initial_phase_artifacts: dict = {}
    if bootstrap_character_seed_nodes or bootstrap_environment_seed_nodes:
        initial_phase_artifacts = {
            "reason_blueprint": {
                "proposals": bootstrap_proposals,
                "bootstrap_seed": {
                    "characters": deepcopy(bootstrap_character_seed_nodes),
                    "environments": deepcopy(bootstrap_environment_seed_nodes),
                },
            }
        }

    previous_last_chapter = _previous_event_last_chapter(preevent)
    recent_plot_refs = (
        previous_last_chapter.get("refs", {}).get("plots", [])
        if isinstance(previous_last_chapter, dict)
        else []
    )
    global_unresolved_threads = (
        _previous_event_unresolved_threads(preevent, premise_id)
        if enable_foreshadowing_injection
        else []
    )
    bounded_plot_refs, previous_unresolved_threads = _select_bounded_plot_inheritance(
        recent_plot_refs=recent_plot_refs,
        inherited_plot_refs=inherited.get("plots", []) or [],
        unresolved_plot_refs=global_unresolved_threads,
        limit=MAX_EVENT_ACTIVE_PLOT_REFS,
    )
    bootstrap_result["plots"] = bounded_plot_refs

    previous_event_summary = _previous_event_summary(preevent)
    trace.log(
        "event_planning_start",
        agent="EventPlanner",
        phase="reason_inquire",
        event_index=event_index,
        content=one_event_abstract,
        payload={
            "first_event": first_event,
            "bootstrap_result": bootstrap_result,
            "true_protagonist_refs": true_protagonist_refs,
            "previous_event_summary": previous_event_summary,
            "inherited_active_refs": inherited,
            "global_unresolved_thread_count": len(global_unresolved_threads),
            "previous_unresolved_threads": previous_unresolved_threads,
            "enable_foreshadowing_injection": enable_foreshadowing_injection,
        },
    )

    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    class OverallState(TypedDict):
        ca_state: AgentState
        ea_state: AgentState
        pa_state: AgentState
        pla_state: AgentState
        cra_state: AgentState
        next_step: str
        pending_tasks: List[dict]
        current_tool_call_id: str
        current_phase: str
        phase_artifacts: dict
        ca_replies_received: int
        ea_replies_received: int
        pa_replies_received: int
        phase_step_count: int
        phase_nudge_sent: bool
        revision_done: bool

    @tool
    def discuss_with_ca(question: str) -> str:
        """
        Ask the Character Manager about character states, relationships, goals, abilities, function-slot conflicts, etc.
        """
        print(f">>> [EventPlanner Tool] -> CharacterManager: '{question}'")
        trace.log(
            "agent_handoff",
            agent="EventPlanner",
            to_agent="CharacterManager",
            phase="reason_inquire",
            event_index=event_index,
            content=question,
        )
        return "CharacterManager has received the question and is preparing a reply."

    @tool
    def discuss_with_ea(question: str) -> str:
        """
        Ask the Environment Manager about physical-space or conceptual-rule information in T_E.
        """
        print(f">>> [EventPlanner Tool] -> EnvironmentManager: '{question}'")
        trace.log(
            "agent_handoff",
            agent="EventPlanner",
            to_agent="EnvironmentManager",
            phase="reason_inquire",
            event_index=event_index,
            content=question,
        )
        return "EnvironmentManager has received the question and is preparing a reply."

    @tool
    def discuss_with_pa(question: str) -> str:
        """
        Ask the Plot Manager about historical plot nodes in C_P, open foreshadowing, causal chains, etc.
        """
        print(f">>> [EventPlanner Tool] -> PlotManager: '{question}'")
        trace.log(
            "agent_handoff",
            agent="EventPlanner",
            to_agent="PlotManager",
            phase="reason_inquire",
            event_index=event_index,
            content=question,
        )
        return "PlotManager has received the question and is preparing a reply."

    @tool
    def request_character_proposal(requirement_json: str) -> str:
        """
        Create a new character. requirement_json MUST copy the EventBlueprint's exact
        char_req_NN requirement_key; missing/malformed keys are rejected before any LLM call.
        """
        requirement = _parse_requirement_payload(requirement_json)
        try:
            requirement_key = _require_proposal_key(requirement, "char_req_")
        except ValueError as exc:
            result = {
                "type": "proposal_request_error",
                "requirement_key": requirement.get("requirement_key", ""),
                "temp_id": None,
                "error": str(exc),
            }
            trace.log(
                "entity_proposal_rejected",
                agent="CharacterManager",
                to_agent="EventPlanner",
                phase="reason_blueprint",
                event_index=event_index,
                content=result,
                payload=result,
                status="error",
            )
            return json.dumps(result, ensure_ascii=False)
        proposal = propose_character_creation(
            premise_id=premise_id,
            requirement=requirement,
            premise=premise,
            event_context=one_event_abstract,
        )
        result = {
            "type": "character_create_proposal",
            "requirement_key": requirement_key,
            "temp_id": _temp_id_from_requirement(requirement, CHARACTER_TEMP_PREFIX),
            "proposal": proposal,
        }
        trace.log(
            "entity_proposal",
            agent="CharacterManager",
            to_agent="EventPlanner",
            phase="reason_blueprint",
            event_index=event_index,
            content=result,
            payload=result,
        )
        return json.dumps(result, ensure_ascii=False)

    @tool
    def request_environment_proposal(requirement_json: str) -> str:
        """
        Create a new environment node (T_phys or T_conc): submit a requirement to EnvironmentManager.
        requirement_json MUST copy the EventBlueprint's exact env_req_NN requirement_key;
        it MUST also preserve branch and parent_hint (null, e_K, or e_tmp_NN).
        """
        requirement = _parse_requirement_payload(requirement_json)
        try:
            requirement = _normalize_environment_proposal_requirement(requirement)
            requirement_key = requirement["requirement_key"]
        except ValueError as exc:
            result = {
                "type": "proposal_request_error",
                "requirement_key": requirement.get("requirement_key", ""),
                "temp_id": None,
                "error": str(exc),
            }
            trace.log(
                "entity_proposal_rejected",
                agent="EnvironmentManager",
                to_agent="EventPlanner",
                phase="reason_blueprint",
                event_index=event_index,
                content=result,
                payload=result,
                status="error",
            )
            return json.dumps(result, ensure_ascii=False)
        proposal = propose_environment_creation(
            premise_id=premise_id,
            requirement=requirement,
            premise=premise,
            event_context=one_event_abstract,
        )
        result = {
            "type": "environment_create_proposal",
            "requirement_key": requirement_key,
            "temp_id": _temp_id_from_requirement(requirement, ENVIRONMENT_TEMP_PREFIX),
            "requirement": deepcopy(requirement),
            "proposal": proposal,
        }
        trace.log(
            "entity_proposal",
            agent="EnvironmentManager",
            to_agent="EventPlanner",
            phase="reason_blueprint",
            event_index=event_index,
            content=result,
            payload=result,
        )
        return json.dumps(result, ensure_ascii=False)

    event_output_contract = f"""
# Final Event Output Protocol (output only in the reason_finalize phase)
You must output an Event JSON that conforms to the structure below. Do not output any extra explanatory text.

## ID Rules
1. refs.characters / refs.environments / refs.plots may only contain string ids.
2. Formal character ids are `c_K`, formal environment ids are `e_K`, formal plot ids are `p_K`.
3. A new character may only use a proposal-backed temp_id (c_tmp_NN) obtained from `request_character_proposal` or supplied above as the system bootstrap c_tmp_00.
4. A new environment may only use a proposal-backed temp_id (e_tmp_NN) obtained from `request_environment_proposal` or supplied above as the system bootstrap e_tmp_00.
5. Only the true protagonist ref(s) ({json.dumps(true_protagonist_refs, ensure_ascii=False)}) must appear in every chapter's refs.characters. Other inherited seed characters are supporting context and belong only in chapters where they are active.

## Top-level structure
{{
  "event_meta": {{
    "event_id": "event_01",
    "title": "Event title",
    "abstract": "Brief summary of the Event",
    "event_goal": "Overall goal of the Event",
    "core_conflict": "Core conflict",
    "chapter_count": 2,
    "entry_state": "Opening state of the Event",
    "exit_hook": "Hook at the end of the Event"
  }},
  "planning_state": {{
    "assumptions": [],
    "open_questions": [],
    "info_sufficiency": {{
      "status": "complete",
      "missing_must_have": []
    }}
  }},
  "seed_subgraph": {{
    "characters": {{
      "existing": {json.dumps(bootstrap_existing_characters, ensure_ascii=False)},
      "new": {json.dumps(bootstrap_character_seed_nodes, ensure_ascii=False)},
      "relationships": []
    }},
    "environments": {{
      "existing": {json.dumps(bootstrap_existing_environments, ensure_ascii=False)},
      "new": {json.dumps(bootstrap_environment_seed_nodes, ensure_ascii=False)}
    }},
    "plots": {{
      "existing": {json.dumps(bootstrap_result.get("plots", []), ensure_ascii=False)},
      "unresolved_threads": []
    }}
  }},
  "chapters": [
    {{
      "chapter_no": 1,
      "pure_plot": "Pure-plot summary of this chapter",
      "chapter_goal": "Goal of this chapter",
      "chapter_conflict": "Conflict of this chapter",
      "chapter_turn": "Turn of this chapter",
      "chapter_hook_out": "Hook at the end of this chapter",
      "refs": {{ "characters": [], "environments": [], "plots": [] }},
      "expected_deltas": {{ "character_updates": [], "environment_updates": [], "plot_updates": [] }}
    }},
    {{
      "chapter_no": 2,
      "pure_plot": "Pure-plot summary of this chapter",
      "chapter_goal": "Goal of this chapter",
      "chapter_conflict": "Conflict of this chapter",
      "chapter_turn": "Turn of this chapter",
      "chapter_hook_out": "Hook at the end of this chapter",
      "refs": {{ "characters": [], "environments": [], "plots": [] }},
      "expected_deltas": {{ "character_updates": [], "environment_updates": [], "plot_updates": [] }}
    }}
  ]
}}

## Hard Requirements
1. Each Event must contain exactly 2 chapters.
2. Each chapter must have pure_plot / chapter_goal / chapter_conflict / chapter_turn / chapter_hook_out / refs / expected_deltas.
3. planning_state.info_sufficiency.missing_must_have must be empty.
4. The protagonist ref {json.dumps(true_protagonist_refs, ensure_ascii=False)} must appear in every chapter.refs.characters (to prevent protagonist swapping). A system-proposed temp protagonist is valid only when its full proposal-backed node remains registered in seed_subgraph.characters.new.
5. `seed_subgraph.plots.unresolved_threads` may contain only already-canonical `p_K` refs:
   - Carry forward each historical foreshadowing thread that this Event does not plan to resolve.
   - Every unresolved_threads ref must also appear in seed_subgraph.plots.existing.
   - A historical thread planned for payoff must instead appear in the paying-off chapter's refs.plots.
   - Do not invent `p_tmp_*`, and do not list hooks newly planted in this Event. New plot nodes and their foreshadowing markers are created only after prose generation by reverse canon extraction.
6. If a newly proposed character already has a relationship with another active character at Event entry, register it in `seed_subgraph.characters.relationships` using `{{"source":"c_K|c_tmp_NN","target":"c_K|c_tmp_NN","current_type":"...","change_history":[{{"from":"none","to":"...","reason":"..."}}]}}`. Endpoints must be registered refs. Do not pre-apply a relationship change that is supposed to happen inside the generated chapters; put that change in expected_deltas instead.
7. Across `seed_subgraph.plots.existing`, `unresolved_threads`, and every `chapters[*].refs.plots`, the total number of unique active `p_K` refs must not exceed {MAX_EVENT_ACTIVE_PLOT_REFS}.
8. Every `seed_subgraph.environments.new` entry must preserve the proposal requirement's exact `branch` and `parent_hint` as `parent_id`. A temp parent must also be registered in `environments.new`, be proposal-backed, and use the same branch. Use null only to mount directly below the corresponding branch root; never use an environment name as a parent reference.
"""

    base_sys_prompt = f"""
# Role
You are the Event Planner inside Topo-Narrator (the Reason stage of the paper's Plan-Reason-Write framework).
You collaborate with 3 Memory Managers (G_C / C_P / T_E) to refine a single Event abstract into a writable E_t (containing 2 chapters).

# Core Principles
1. You drive narrative / structural decisions; the 3 Managers are assistant agents that you may query repeatedly.
2. Each phase produces only its own phase artifact, but you may query the Managers at any time to fill in missing information.
3. The true protagonist ref must never be swapped out: {json.dumps(true_protagonist_refs, ensure_ascii=False)}. It may be a committed c_K or the system-proposed c_tmp_00 for a fresh story. Inherited supporting-character refs are not protagonists and must not be broadcast into unrelated chapters.
4. Context engineering: ask CharacterManager only about characters, EnvironmentManager only about T_E,
   and PlotManager only about C_P. Do not indiscriminately dump the whole book to every agent.

# Current Task
Novel premise:
{premise}

Current Event Abstract:
{one_event_abstract}

Is this the first Event: {first_event}
Previous Event summary: {previous_event_summary if previous_event_summary else "None"}
Open foreshadowing from the previous Event (plot refs): {json.dumps(previous_unresolved_threads, ensure_ascii=False) if previous_unresolved_threads else "None"}
{"  → These threads must be handled explicitly in this Event: either resolve them in one of the chapters (place the corresponding plot ref into that chapter's refs.plots and pay it off explicitly in pure_plot), or continue them into the next Event (keep them in seed_subgraph.plots.unresolved_threads and explain why they are being extended). Silently dropping them is forbidden." if previous_unresolved_threads else ""}

Grounded seed context (contains true protagonist plus bounded inherited supporting context):
{json.dumps(bootstrap_result, ensure_ascii=False)}

System-proposed Event-local seed nodes (already proposal-backed; reuse them and do not ask a Manager to look them up in M_global):
characters.new = {json.dumps(bootstrap_character_seed_nodes, ensure_ascii=False)}
environments.new = {json.dumps(bootstrap_environment_seed_nodes, ensure_ascii=False)}

True protagonist refs (the only character refs mandatory in every chapter):
{json.dumps(true_protagonist_refs, ensure_ascii=False)}

# Global ID Rules
1. refs.{{characters,environments,plots}} may only contain string ids.
2. Formal ids are `c_K`/`e_K`/`p_K`. Only characters and environments may use proposal-backed temp ids (`c_tmp_NN`/`e_tmp_NN`); plot temp ids do not exist in Event planning. The system-proposed c_tmp_00/e_tmp_00 above are already valid proposals and must be registered under seed_subgraph.*.new if referenced.
3. Never invent your own formal ids; never mix in names / integers / natural language.

# Phase Flow (3 phases, injected by the system via HumanMessage)
1. reason_inquire    : call discuss_with_* tools to query Memory Managers (no separate tag artifact)
2. reason_blueprint  : output EventBlueprint and call character/environment proposal tools as needed
3. reason_finalize   : output the final Event JSON (exactly 2 chapters)

Stay focused on the current phase. Do not do the work of the next phase early; the final Event JSON may only be output in the reason_finalize phase.
"""

    PHASE_PROMPTS: dict[str, str] = {
        "reason_inquire": f"""
[System Dispatch] Entering reason_inquire (consult Memory Managers).

# Goal of this phase
Decide what information you still need in order to plan this Event, then **directly call**
discuss_with_ca / discuss_with_ea / discuss_with_pa to ask the corresponding Memory Manager.

The Managers only see the questions you actually pass via tool_calls — there is no separate
tag, JSON channel, or sidecar artifact. Each non-empty query must be sent as its own
discuss_with_* tool call (you may dispatch several in the same turn via parallel tool_calls).
Each question must be one focused, decision-oriented retrieval request and at most
{MAX_MANAGER_QUESTION_CHARS} characters. Split broad multi-scene or multi-entity audits into
later follow-ups; an overlong question is rejected before dispatch.

# Advancement condition
Before reason_blueprint, obtain at least one focused reply from **each** Manager:
CharacterManager, EnvironmentManager, and PlotManager. The system advances only after all
three replies have returned and no dispatch is pending. Outputting prose or JSON without
calling all three tools does **NOT** advance the phase.

# Anti-pattern (DO NOT DO THIS)
- ❌ Reasoning out loud "I should ask the Character Manager about X" without invoking the tool.
- ❌ Emitting a `tool_uses` / `<|multi_tool_use.parallel|>` block inside the content. If you want
  parallel calls, use the API's native parallel tool_calls mechanism (multiple tool_calls in
  one turn). Putting that wrapper into content makes the call invisible and wastes turns.
- ❌ Outputting summary JSON of the questions instead of issuing the tool calls.

# Minimum viable progress per turn
Prefer dispatching discuss_with_ca, discuss_with_ea, and discuss_with_pa in parallel. If one
is still missing on a later turn, call that Manager with a focused consistency question.
""",

        "reason_blueprint": f"""
[System Dispatch] Entering reason_blueprint (skeleton + entity proposals).

# Goals of this phase
- Based on the retrieval results from the previous phase, design the EventBlueprint with exactly 2 chapters.
- If new characters / environments are needed, **directly** call request_character_proposal /
  request_environment_proposal to obtain temp_ids (no extra approval is required).
- Every proposal tool's requirement_json must copy the matching requirement object from this
  blueprint, including its exact requirement_key (char_req_NN / env_req_NN). A missing or
  malformed key is rejected before an LLM call and does not satisfy the requirement.
- c_tmp_00/e_tmp_00, when present in the system-proposed seed, already have complete
  proposals. Reuse them and do not create duplicate protagonist/opening-scene proposals.

# Output format
<{EVENT_BLUEPRINT_TAG}>
{{
  "event_goal": "...",
  "core_conflict": "...",
  "chapter_count": 2,
  "beats": [
    {{"chapter_no": 1, "responsibility": "The narrative responsibility this chapter carries inside the Event"}},
    {{"chapter_no": 2, "responsibility": "The narrative responsibility this chapter carries inside the Event"}}
  ],
  "preliminary_refs": {{
    "characters": {json.dumps(bootstrap_result.get("characters", []), ensure_ascii=False)},
    "environments": {json.dumps(bootstrap_result.get("environments", []), ensure_ascii=False)},
    "plots": {json.dumps(bootstrap_result.get("plots", []), ensure_ascii=False)}
  }},
  "needed_new_characters": [
    {{"requirement_key": "char_req_01", "function_slot": "...", "reason": "..."}}
  ],
  "needed_new_environments": [
    {{"requirement_key": "env_req_01", "branch": "phys|conc", "parent_hint": "e_K|e_tmp_NN|null", "function_slot": "...", "reason": "..."}}
  ],
  "assumptions": [],
  "open_questions": []
}}
</{EVENT_BLUEPRINT_TAG}>

After outputting the EventBlueprint, call the corresponding proposal tool exactly once for each
needed_new_character / needed_new_environment (request_character_proposal / request_environment_proposal).
For every environment, parent_hint is a topology decision: use a canonical existing e_K ref, an
already available/proposed e_tmp_NN ref in the same branch, or null for the corresponding root.
Copy it unchanged into request_environment_proposal; never replace it with a node name.
Never pre-propose a plot node: new plots are created exclusively from written prose by reverse canon extraction.
Once all proposals have returned, the system will advance to reason_finalize.
""",

        "reason_finalize": event_output_contract + f"""

[System Dispatch] Entering reason_finalize (chapter-level finalization).

# Goal of this phase
Turn the EventBlueprint + the obtained temp_ids into the final Event JSON (exactly 2 chapters).

# Hard reminders
1. The protagonist ref must appear in every chapter's refs.characters.
2. Every chapter's refs.environments must contain at least 1 e_K / e_tmp_NN (scene is mandatory, scene-anchored).
3. planning_state.info_sufficiency.missing_must_have must be empty.
4. **Every c_tmp_NN / e_tmp_NN appearing in chapter refs MUST also appear in seed_subgraph.{{characters|environments}}.new[*].temp_id with full schema fields.** Every existing ID (c_K / e_K / p_K) in chapter refs must be registered in the matching existing list. `p_tmp_*` is forbidden.
{"5. unresolved_threads may contain only historical p_K refs that are also registered in plots.existing. Carry forward open threads from " + json.dumps(previous_unresolved_threads, ensure_ascii=False) + " that this Event does not plan to resolve; put a thread planned for payoff in the relevant chapter refs.plots. Do not add newly planned hooks here." if enable_foreshadowing_injection else "5. unresolved_threads may contain only existing p_K refs also registered in plots.existing; p_tmp_* is forbidden (ablation group may leave the list empty)."}
6. The union of seed_subgraph.plots.existing, unresolved_threads, and all chapter refs.plots must contain at most {MAX_EVENT_ACTIVE_PLOT_REFS} unique p_K refs.
""",
    }

    sys_prompt = base_sys_prompt
    plan_model = build_langchain_model_client()
    pla = plan_model.bind_tools(
        [
            discuss_with_ca,
            discuss_with_ea,
            discuss_with_pa,
            request_character_proposal,
            request_environment_proposal,
        ]
    )

    def _evaluate_phase_transition(state: "OverallState", last_message_content: str):
        current = state.get("current_phase", "reason_inquire")

        if current == "reason_inquire":
            all_managers_replied = not _missing_reason_manager_replies(state)
            if all_managers_replied and not state.get("pending_tasks"):
                return "reason_blueprint", None
            return None, None

        if current == "reason_blueprint":
            blueprint = _extract_tagged_json(last_message_content, EVENT_BLUEPRINT_TAG)
            artifact = {"event_blueprint": blueprint} if blueprint else None
            recorded = state.get("phase_artifacts", {}).get("reason_blueprint", {})
            current_blueprint = blueprint or recorded.get("event_blueprint", {}) or {}
            proposals = recorded.get("proposals", {})
            diagnostics = _blueprint_gate_diagnostics(
                current_blueprint,
                proposals,
                state.get("pending_tasks"),
            )
            if not diagnostics:
                return "reason_finalize", artifact
            return None, artifact

        return None, None

    PHASE_SOFT_NUDGE_AT = 8
    PHASE_FORCE_ADVANCE_AT = 12

    def pla_node(state: OverallState) -> dict:
        current_messages = state["pla_state"]["messages"]

        if state.get("pending_tasks"):
            task_to_dispatch = state["pending_tasks"][0]
            remaining_tasks = state["pending_tasks"][1:]
            agent_name = task_to_dispatch["agent_name"]
            question = task_to_dispatch["question"]
            tool_call_id = task_to_dispatch["tool_call_id"]
            worker_state_key = {
                "Character_Agent": "ca_state",
                "Environment_Agent": "ea_state",
                "Plot_Agent": "pa_state",
            }[agent_name]

            current_worker_msgs = state[worker_state_key]["messages"]
            updated_worker_msgs = current_worker_msgs + [HumanMessage(content=question)]
            return {
                worker_state_key: {"messages": updated_worker_msgs},
                "next_step": agent_name,
                "pending_tasks": remaining_tasks,
                "current_tool_call_id": tool_call_id,
            }

        if state.get("current_phase", "reason_inquire") == "reason_inquire":
            pre_action, pre_missing = _reason_inquire_stall_action(
                state,
                int(state.get("phase_step_count", 0)),
                PHASE_SOFT_NUDGE_AT,
                PHASE_FORCE_ADVANCE_AT,
            )
            if pre_action == "nudge":
                missing_tools = [tool_name for _, tool_name in pre_missing]
                trace.log(
                    "phase_soft_nudge",
                    agent="System",
                    to_agent="EventPlanner",
                    phase="reason_inquire",
                    event_index=event_index,
                    content="Require calls to Manager tools that have not yet replied",
                    payload={
                        "step_count": state.get("phase_step_count"),
                        "missing_tools": missing_tools,
                    },
                    status="revise",
                )
                return {
                    "pla_state": {
                        "messages": current_messages
                        + [HumanMessage(content=_reason_inquire_nudge_message(pre_missing))]
                    },
                    "next_step": "Plan_Agent",
                    "current_phase": "reason_inquire",
                    "phase_step_count": int(state.get("phase_step_count", 0)),
                    "phase_nudge_sent": True,
                }
            if pre_action == "fail":
                missing_tools = [tool_name for _, tool_name in pre_missing]
                error_message = (
                    "reason_inquire exceeded its step limit before all Memory Managers replied; "
                    f"missing tool replies: {missing_tools}"
                )
                trace.log(
                    "error",
                    agent="System",
                    phase="reason_inquire",
                    event_index=event_index,
                    content=error_message,
                    payload={"step_count": state.get("phase_step_count"), "missing_tools": missing_tools},
                    status="error",
                )
                raise RuntimeError(error_message)

        import time as _time
        _t0 = _time.time()
        _phase_now = state.get("current_phase", "reason_inquire")
        print(f"⏳ [EventPlanner|{_phase_now}|step{state.get('phase_step_count', 0)}] Calling LLM... (messages={len(current_messages)})", flush=True)
        result_message = pla.invoke(current_messages)
        print(f"✅ [EventPlanner|{_phase_now}] LLM responded in {_time.time() - _t0:.1f}s", flush=True)

        spoof_rewired = False
        if not getattr(result_message, "tool_calls", None):
            _spoof_calls = _extract_spoofed_tool_calls(str(result_message.content))
            if _spoof_calls:
                result_message = AIMessage(
                    content=str(result_message.content),
                    tool_calls=_spoof_calls,
                )
                spoof_rewired = True

        updated_messages = current_messages + [result_message]
        trace.log(
            "agent_decision",
            agent="EventPlanner",
            phase=state.get("current_phase"),
            event_index=event_index,
            content=str(result_message.content),
            payload={
                "tool_calls": getattr(result_message, "tool_calls", []),
                "message_type": result_message.__class__.__name__,
                "llm_latency_sec": round(_time.time() - _t0, 1),
                "spoof_rewired": spoof_rewired,
            },
        )

        result_content = str(result_message.content)
        current_phase = state.get("current_phase", "reason_inquire")
        phase_artifacts = deepcopy(state.get("phase_artifacts", {})) or {}

        if result_message.tool_calls:
            new_tasks = []
            immediate_feedback = []

            for tool_call in result_message.tool_calls:
                tool_name = tool_call["name"]
                tool_call_id = tool_call["id"]
                args = tool_call.get("args", {})
                trace.log(
                    "agent_tool_call",
                    agent="EventPlanner",
                    phase=current_phase,
                    event_index=event_index,
                    content=tool_name,
                    payload={"tool_call_id": tool_call_id, "tool_name": tool_name, "args": args},
                )

                if tool_name in {
                    "discuss_with_ca",
                    "discuss_with_ea",
                    "discuss_with_pa",
                }:
                    agent_for_tool = {
                        "discuss_with_ca": "Character_Agent",
                        "discuss_with_ea": "Environment_Agent",
                        "discuss_with_pa": "Plot_Agent",
                    }[tool_name]
                    question = args.get("question")
                    question_error = _manager_question_error(question)
                    if question_error:
                        immediate_feedback.append(
                            ToolMessage(
                                content=(
                                    f"[Manager query rejected before dispatch] {question_error}"
                                ),
                                tool_call_id=tool_call_id,
                            )
                        )
                        trace.log(
                            "manager_query_rejected",
                            agent="System",
                            to_agent="EventPlanner",
                            phase=current_phase,
                            event_index=event_index,
                            content=question_error,
                            payload={
                                "tool_name": tool_name,
                                "tool_call_id": tool_call_id,
                                "question_length": len(question) if isinstance(question, str) else None,
                            },
                            status="revise",
                        )
                    else:
                        new_tasks.append({
                            "agent_name": agent_for_tool,
                            "question": question,
                            "tool_call_id": tool_call_id,
                        })
                elif tool_name == "request_character_proposal":
                    payload = request_character_proposal.invoke(args)
                    try:
                        parsed = json.loads(payload)
                        cache_key = parsed.get("temp_id")
                        if parsed.get("type") == "character_create_proposal" and cache_key:
                            phase_artifacts \
                                .setdefault("reason_blueprint", {}) \
                                .setdefault("proposals", {}) \
                                .setdefault("characters", {})[cache_key] = parsed
                    except Exception:
                        pass
                    immediate_feedback.append(
                        ToolMessage(content=payload, tool_call_id=tool_call_id)
                    )
                elif tool_name == "request_environment_proposal":
                    payload = request_environment_proposal.invoke(args)
                    try:
                        parsed = json.loads(payload)
                        cache_key = parsed.get("temp_id")
                        if parsed.get("type") == "environment_create_proposal" and cache_key:
                            phase_artifacts \
                                .setdefault("reason_blueprint", {}) \
                                .setdefault("proposals", {}) \
                                .setdefault("environments", {})[cache_key] = parsed
                    except Exception:
                        pass
                    immediate_feedback.append(
                        ToolMessage(content=payload, tool_call_id=tool_call_id)
                    )
            if immediate_feedback:
                updated_messages = updated_messages + immediate_feedback

            if current_phase == "reason_blueprint":
                bp = _extract_tagged_json(result_content, EVENT_BLUEPRINT_TAG)
                if bp:
                    blueprint_record = phase_artifacts.setdefault("reason_blueprint", {})
                    existing_bp = blueprint_record.get("event_blueprint")
                    existing_is_well_formed = bool(
                        isinstance(existing_bp, dict)
                        and _blueprint_requirements_are_well_formed(
                            existing_bp.get("needed_new_characters", []), "char_req_"
                        )
                        and _blueprint_requirements_are_well_formed(
                            existing_bp.get("needed_new_environments", []), "env_req_"
                        )
                    )
                    if existing_is_well_formed and bp != existing_bp:
                        # Once a valid blueprint exists its requirement keys are
                        # authoritative. Replacing it after proposals return makes
                        # the cache impossible to satisfy and caused the observed
                        # reason_blueprint step-limit failure.
                        updated_messages.append(HumanMessage(content=(
                            "[System Contract Check] The valid EventBlueprint is already frozen. "
                            "Do not redefine its requirements; request only the still-missing "
                            "proposals for the frozen requirement_key values."
                        )))
                        trace.log(
                            "blueprint_mutation_rejected",
                            agent="System",
                            to_agent="EventPlanner",
                            phase="reason_blueprint",
                            event_index=event_index,
                            content="Reject changes to the frozen blueprint during the proposal workflow",
                            payload={"existing_blueprint_retained": True},
                            status="revise",
                        )
                    else:
                        blueprint_record["event_blueprint"] = bp
                        trace.log(
                            "phase_artifact",
                            agent="EventPlanner",
                            phase="reason_blueprint",
                            event_index=event_index,
                            content="reason_blueprint artifact recorded in the same turn as tool_calls",
                            payload={"artifact_keys": ["event_blueprint"]},
                            status="ok",
                        )

            if current_phase == "reason_inquire" and new_tasks:
                dispatched = phase_artifacts.setdefault("reason_inquire", {}).setdefault(
                    "dispatched", {"ca": 0, "ea": 0, "pa": 0}
                )
                for task in new_tasks:
                    agent_short = {
                        "Character_Agent": "ca",
                        "Environment_Agent": "ea",
                        "Plot_Agent": "pa",
                    }.get(task.get("agent_name"))
                    if agent_short:
                        dispatched[agent_short] = dispatched.get(agent_short, 0) + 1

            step_count = int(state.get("phase_step_count", 0)) + 1

            # Proposal tools are synchronous. If the frozen blueprint and every
            # proposal are now complete, transition deterministically in this
            # same graph step instead of requiring the LLM to emit an additional
            # no-tool message (the source of the observed 12-step dead loop).
            if current_phase == "reason_blueprint" and not new_tasks:
                transition_state = dict(state)
                transition_state["phase_artifacts"] = phase_artifacts
                transition_state["pending_tasks"] = []
                next_phase, _ = _evaluate_phase_transition(transition_state, "")
                if next_phase == "reason_finalize":
                    prev_summary = _summarize_artifact(
                        current_phase,
                        phase_artifacts.get(current_phase, {}),
                    )
                    updated_messages.append(HumanMessage(content=(
                        f"[System Dispatch] The previous phase ({current_phase}) artifact is confirmed:\n"
                        f"{prev_summary}\n\n{PHASE_PROMPTS[next_phase]}"
                    )))
                    trace.log(
                        "phase_transition",
                        agent="System",
                        to_agent="EventPlanner",
                        phase=next_phase,
                        event_index=event_index,
                        content=f"{current_phase} -> {next_phase}",
                        payload={
                            "prev_phase": current_phase,
                            "next_phase": next_phase,
                            "summary": prev_summary,
                            "deterministic_after_proposals": True,
                        },
                        status="ok",
                    )
                    return {
                        "pla_state": {"messages": updated_messages},
                        "pending_tasks": [],
                        "next_step": "Plan_Agent",
                        "current_phase": next_phase,
                        "phase_artifacts": phase_artifacts,
                        "phase_step_count": 0,
                        "phase_nudge_sent": False,
                    }

                blueprint_record = phase_artifacts.get("reason_blueprint", {})
                diagnostics = _blueprint_gate_diagnostics(
                    blueprint_record.get("event_blueprint"),
                    blueprint_record.get("proposals", {}),
                    [],
                )
                if diagnostics:
                    updated_messages.append(HumanMessage(content=(
                        "[System Contract Check] reason_blueprint cannot advance yet:\n- "
                        + "\n- ".join(diagnostics)
                    )))
                    trace.log(
                        "blueprint_contract_blocked",
                        agent="System",
                        to_agent="EventPlanner",
                        phase="reason_blueprint",
                        event_index=event_index,
                        content="reason_blueprint gate not satisfied",
                        payload={"step_count": step_count, "diagnostics": diagnostics},
                        status="revise",
                    )
                    if step_count >= PHASE_FORCE_ADVANCE_AT:
                        raise RuntimeError(
                            "reason_blueprint exceeded its step limit; unmet contract: "
                            + "; ".join(diagnostics)
                        )
            return {
                "pla_state": {"messages": updated_messages},
                "pending_tasks": new_tasks,
                "next_step": "Plan_Agent",
                "current_phase": current_phase,
                "phase_artifacts": phase_artifacts,
                "phase_step_count": step_count,
            }

        candidate_event = _try_parse_event_candidate(result_content)
        if candidate_event and current_phase == "reason_finalize":
            trace.log(
                "event_planning_done",
                agent="EventPlanner",
                to_agent="Validator",
                phase="reason_finalize",
                event_index=event_index,
                content=result_content,
                payload=candidate_event,
            )
            return {
                "pla_state": {"messages": updated_messages},
                "next_step": "Critic_Agent",
                "current_phase": current_phase,
                "phase_artifacts": phase_artifacts,
            }

        next_phase, artifact = _evaluate_phase_transition(state, result_content)
        if artifact:
            phase_artifacts.setdefault(current_phase, {}).update(artifact)
            trace.log(
                "phase_artifact",
                agent="EventPlanner",
                phase=current_phase,
                event_index=event_index,
                content=f"{current_phase} artifact recorded",
                payload={"artifact_keys": list(artifact.keys())},
                status="ok",
            )

        if next_phase:
            prev_summary = _summarize_artifact(current_phase, phase_artifacts.get(current_phase, {}))
            transition_msg = HumanMessage(
                content=(
                    f"[System Dispatch] The previous phase ({current_phase}) artifact is confirmed:\n"
                    f"{prev_summary}\n\n"
                    + PHASE_PROMPTS[next_phase]
                )
            )
            updated_messages = updated_messages + [transition_msg]
            trace.log(
                "phase_transition",
                agent="System",
                to_agent="EventPlanner",
                phase=next_phase,
                event_index=event_index,
                content=f"{current_phase} -> {next_phase}",
                payload={"prev_phase": current_phase, "next_phase": next_phase, "summary": prev_summary},
                status="ok",
            )
            return {
                "pla_state": {"messages": updated_messages},
                "next_step": "Plan_Agent",
                "current_phase": next_phase,
                "phase_artifacts": phase_artifacts,
                "phase_step_count": 0,
                "phase_nudge_sent": False,
            }

        step_count = int(state.get("phase_step_count", 0)) + 1
        nudge_sent = bool(state.get("phase_nudge_sent", False))

        if current_phase == "reason_inquire":
            inquire_action, missing_replies = _reason_inquire_stall_action(
                state,
                step_count,
                PHASE_SOFT_NUDGE_AT,
                PHASE_FORCE_ADVANCE_AT,
            )
            if inquire_action == "fail":
                missing_tools = [tool_name for _, tool_name in missing_replies]
                error_message = (
                    "reason_inquire exceeded its step limit before all Memory Managers replied; "
                    f"missing tool replies: {missing_tools}"
                )
                trace.log(
                    "error",
                    agent="System",
                    phase="reason_inquire",
                    event_index=event_index,
                    content=error_message,
                    payload={"step_count": step_count, "missing_tools": missing_tools},
                    status="error",
                )
                raise RuntimeError(error_message)
            if inquire_action == "nudge":
                updated_messages = updated_messages + [
                    HumanMessage(content=_reason_inquire_nudge_message(missing_replies))
                ]
                nudge_sent = True
                trace.log(
                    "phase_soft_nudge",
                    agent="System",
                    to_agent="EventPlanner",
                    phase="reason_inquire",
                    event_index=event_index,
                    content="Require calls to Manager tools that have not yet replied",
                    payload={
                        "step_count": step_count,
                        "missing_tools": [tool_name for _, tool_name in missing_replies],
                    },
                    status="revise",
                )
            return {
                "pla_state": {"messages": updated_messages},
                "next_step": "Plan_Agent",
                "current_phase": current_phase,
                "phase_artifacts": phase_artifacts,
                "phase_step_count": step_count,
                "phase_nudge_sent": nudge_sent,
            }


        blueprint_diagnostics: list[str] = []
        if current_phase == "reason_blueprint":
            blueprint_record = phase_artifacts.get("reason_blueprint", {})
            blueprint_diagnostics = _blueprint_gate_diagnostics(
                blueprint_record.get("event_blueprint"),
                blueprint_record.get("proposals", {}),
                state.get("pending_tasks"),
            )

        if step_count >= PHASE_FORCE_ADVANCE_AT:
            detail = (
                "; ".join(blueprint_diagnostics)
                if blueprint_diagnostics
                else "required artifact/tool replies remain incomplete"
            )
            trace.log(
                "phase_contract_failed",
                agent="System",
                to_agent="EventPlanner",
                phase=current_phase,
                event_index=event_index,
                content=f"{current_phase} reached the step limit without satisfying the contract",
                payload={"step_count": step_count, "diagnostics": blueprint_diagnostics},
                status="error",
            )
            raise RuntimeError(
                f"{current_phase} exceeded its step limit without satisfying the phase contract: "
                f"{detail}"
            )

        if step_count >= PHASE_SOFT_NUDGE_AT and not nudge_sent:
            diagnostic_text = (
                "\nUnmet contract items:\n- " + "\n- ".join(blueprint_diagnostics)
                if blueprint_diagnostics
                else ""
            )
            nudge = HumanMessage(content=(
                f"[System Dispatch] {current_phase} has run multiple rounds without producing this phase's artifact. "
                f"Stop asking follow-up questions or expanding requirements immediately, and output the tagged JSON artifact for this phase right now."
                f"{diagnostic_text}"
            ))
            updated_messages = updated_messages + [nudge]
            nudge_sent = True
            trace.log(
                "phase_soft_nudge",
                agent="System",
                to_agent="EventPlanner",
                phase=current_phase,
                event_index=event_index,
                content="Send a soft nudge",
                payload={"step_count": step_count, "diagnostics": blueprint_diagnostics},
                status="revise",
            )

        return {
            "pla_state": {"messages": updated_messages},
            "next_step": "Plan_Agent",
            "current_phase": current_phase,
            "phase_artifacts": phase_artifacts,
            "phase_step_count": step_count,
            "phase_nudge_sent": nudge_sent,
        }

    def worker_node(state: OverallState, agent_name: str, agent_runnable) -> dict:
        worker_state_key = {
            "Character_Agent": "ca_state",
            "Environment_Agent": "ea_state",
            "Plot_Agent": "pa_state",
        }[agent_name]

        tool_call_id = state.get("current_tool_call_id")
        prior_worker_msgs = state[worker_state_key]["messages"]
        result = agent_runnable.invoke(state[worker_state_key])
        updated_worker_msgs = result.get("messages", []) if isinstance(result, dict) else []
        answer_content = (
            updated_worker_msgs[-1].content if updated_worker_msgs else ""
        )
        grounded = _manager_reply_is_grounded(
            prior_worker_msgs,
            updated_worker_msgs,
            answer_content,
        )
        trace.log(
            "agent_response",
            agent=agent_name.replace("_", ""),
            to_agent="EventPlanner",
            phase=state.get("current_phase"),
            event_index=event_index,
            content=answer_content,
            payload={"tool_call_id": tool_call_id, "grounded": grounded},
            status="ok" if grounded else "revise",
        )
        reply_prefix = f"[Reply from {agent_name}]" if grounded else (
            f"[UNVERIFIED reply from {agent_name}: no retrieval ToolMessage was produced; "
            "this does not satisfy the Manager gate]"
        )
        feedback_message = ToolMessage(
            content=f"{reply_prefix}\n{answer_content}",
            tool_call_id=tool_call_id,
        )
        updated_pla_msgs = state["pla_state"]["messages"] + [feedback_message]
        reply_counter_key = {
            "Character_Agent": "ca_replies_received",
            "Environment_Agent": "ea_replies_received",
            "Plot_Agent": "pa_replies_received",
        }[agent_name]
        return {
            "pla_state": {"messages": updated_pla_msgs},
            worker_state_key: {"messages": updated_worker_msgs},
            "next_step": "Plan_Agent",
            "current_tool_call_id": None,
            reply_counter_key: state.get(reply_counter_key, 0) + (1 if grounded else 0),
        }

    def critic_node(state: OverallState) -> dict:
        draft_message = state["pla_state"]["messages"][-1]
        draft_content = str(draft_message.content)
        parsed_event = extract_json.extract_json(draft_content)
        parsed_event = event_schema.normalize_event_payload(parsed_event)
        if isinstance(parsed_event, dict):
            # Canonicalize proposal-backed nodes *before* validation.  In
            # particular, this restores the requirement's authoritative parent
            # edge when the finalize LLM copied a model recommendation instead.
            _backfill_seed_subgraph_from_proposals(
                parsed_event,
                state.get("phase_artifacts", {}),
                trace_logger=trace,
                event_index=event_index,
            )
        errors = list(event_schema.validate_event_payload(parsed_event))
        errors.extend(
            _validate_environment_parent_contract(
                parsed_event,
                state.get("phase_artifacts", {}),
                global_environment_contract_graph,
            )
        )
        errors.extend(
            _validate_plot_inheritance_contract(
                parsed_event,
                previous_unresolved_threads,
                MAX_EVENT_ACTIVE_PLOT_REFS,
            )
        )

        protagonist_refs = set(true_protagonist_refs)
        if protagonist_refs and isinstance(parsed_event, dict):
            for ch in parsed_event.get("chapters", []) or []:
                if not isinstance(ch, dict):
                    continue
                ch_chars = set(ch.get("refs", {}).get("characters", []) or [])
                if not protagonist_refs.issubset(ch_chars):
                    errors.append(
                        f"Chapter {ch.get('chapter_no')} refs.characters is missing the protagonist ref "
                        f"{sorted(protagonist_refs - ch_chars)} (protagonist-swap guardrail)"
                    )

        if not errors and isinstance(parsed_event, dict):
            continuity_issues = _continuity_check_via_llm(parsed_event, preevent)
            if continuity_issues:
                errors.extend(continuity_issues)
                trace.log(
                    "continuity_issues_detected",
                    agent="Validator",
                    phase=state.get("current_phase"),
                    event_index=event_index,
                    content=f"Found {len(continuity_issues)} serious cross-chapter contradictions",
                    payload={"issues": continuity_issues},
                    status="revise",
                )

        validator_content = json.dumps({
            "review_summary": {
                "decision": "APPROVE" if not errors else "REVISE",
                "issue_count": len(errors),
            },
            "critical_issues": errors,
        }, ensure_ascii=False, indent=2)
        updated_cra_msgs = state["cra_state"]["messages"] + [AIMessage(content=validator_content)]
        trace.log(
            "validator_result",
            agent="Validator",
            to_agent="EventPlanner",
            phase=state.get("current_phase"),
            event_index=event_index,
            content=validator_content,
            payload={"errors": errors, "decision": "APPROVE" if not errors else "REVISE"},
            status="ok" if not errors else "revise",
        )

        if not errors:
            return {
                "cra_state": {"messages": updated_cra_msgs},
                "next_step": "end",
            }

        if state.get("revision_done"):
            return {
                "cra_state": {"messages": updated_cra_msgs},
                "next_step": "end",
            }
        feedback_msg = HumanMessage(
            content=f"[Final Event structure validation failed] Please fix the issues below and re-output the final Event JSON:\n{validator_content}"
        )
        updated_pla_msgs = state["pla_state"]["messages"] + [feedback_msg]
        return {
            "cra_state": {"messages": updated_cra_msgs},
            "pla_state": {"messages": updated_pla_msgs},
            "next_step": "Plan_Agent",
            "revision_done": True,
        }

    graph = StateGraph(OverallState)
    graph.add_node("Plan_Agent", pla_node)
    graph.add_node("Character_Agent", partial(worker_node, agent_name="Character_Agent", agent_runnable=ca))
    graph.add_node("Environment_Agent", partial(worker_node, agent_name="Environment_Agent", agent_runnable=ea))
    graph.add_node("Plot_Agent", partial(worker_node, agent_name="Plot_Agent", agent_runnable=pa))
    graph.add_node("Critic_Agent", critic_node)

    def router(state: OverallState) -> str:
        return state.get("next_step")

    graph.add_conditional_edges(
        "Plan_Agent",
        router,
        {
            "Character_Agent": "Character_Agent",
            "Environment_Agent": "Environment_Agent",
            "Plot_Agent": "Plot_Agent",
            "Plan_Agent": "Plan_Agent",
            "Critic_Agent": "Critic_Agent",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "Critic_Agent",
        router,
        {
            "Plan_Agent": "Plan_Agent",
            "end": END,
        },
    )
    graph.add_edge("Character_Agent", "Plan_Agent")
    graph.add_edge("Environment_Agent", "Plan_Agent")
    graph.add_edge("Plot_Agent", "Plan_Agent")
    graph.set_entry_point("Plan_Agent")

    app = graph.compile(checkpointer=MemorySaver())
    current_thread_id = str(uuid.uuid4())
    trace.log(
        "agent_decision",
        agent="EventPlanner",
        phase="reason_inquire",
        event_index=event_index,
        content=f"LangGraph thread_id: {current_thread_id}",
        payload={"thread_id": current_thread_id},
    )
    run_config = {"recursion_limit": 400, "configurable": {"thread_id": current_thread_id}}

    state_data = {
        "pla_state": {"messages": [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=one_event_abstract),
            HumanMessage(content=PHASE_PROMPTS["reason_inquire"]),
        ]},
        "ca_state": {"messages": []},
        "ea_state": {"messages": []},
        "pa_state": {"messages": []},
        "cra_state": {"messages": []},
        "next_step": "",
        "pending_tasks": [],
        "phase_artifacts": deepcopy(initial_phase_artifacts),
        "ca_replies_received": 0,
        "ea_replies_received": 0,
        "pa_replies_received": 0,
        "current_tool_call_id": None,
        "current_phase": "reason_inquire",
        "phase_step_count": 0,
        "phase_nudge_sent": False,
    }

    for chunk in app.stream(state_data, config=run_config):
        if not chunk.values():
            continue
        update_dict = list(chunk.values())[0]
        if isinstance(update_dict, dict):
            state_data.update(update_dict)

    final_snapshot = app.get_state(run_config)
    if final_snapshot.values:
        final_content = final_snapshot.values["pla_state"]["messages"][-1].content
        if isinstance(final_content, str) and len(final_content) < 30 and "APPROVE" in final_content:
            final_content = final_snapshot.values["pla_state"]["messages"][-2].content

        try:
            candidate = extract_json.extract_json(str(final_content))
            normalized = event_schema.normalize_event_payload(candidate)
        except Exception as exc:
            raise ValueError("Final Event JSON could not be parsed for topology validation.") from exc
        if not isinstance(normalized, dict):
            raise ValueError("Final Event JSON is not an object during topology validation.")

        _backfill_seed_subgraph_from_proposals(
            normalized,
            final_snapshot.values.get("phase_artifacts", {}),
            trace_logger=trace,
            event_index=event_index,
        )
        topology_errors = _validate_environment_parent_contract(
            normalized,
            final_snapshot.values.get("phase_artifacts", {}),
            global_environment_contract_graph,
        )
        if topology_errors:
            raise ValueError(
                "Final Event environment topology validation failed: "
                + "; ".join(topology_errors)
            )
        final_content = json.dumps(normalized, ensure_ascii=False)

        return final_content
    trace.log(
        "error",
        agent="EventPlanner",
        phase="reason_finalize",
        event_index=event_index,
        content="No state found.",
        status="error",
    )
    return "Error: No state found."
