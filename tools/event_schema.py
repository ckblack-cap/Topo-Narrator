import copy
import json
from typing import Any

from tools.graph_refs import (
    CHARACTER_TEMP_PREFIX,
    ENVIRONMENT_TEMP_PREFIX,
    is_character_ref,
    is_environment_ref,
    is_real_plot_ref,
    make_temp_ref,
    normalize_character_ref,
    normalize_environment_ref,
    normalize_plot_ref,
)


def _is_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_temp_entities(items: list[dict] | None, prefix: str) -> list[dict]:
    normalized: list[dict] = []
    if not isinstance(items, list):
        return normalized

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        node = copy.deepcopy(item)
        node["temp_id"] = node.get("temp_id") or make_temp_ref(prefix, index)
        node["first_appearance_chapter"] = int(node.get("first_appearance_chapter") or 1)
        normalized.append(node)
    return normalized


def _normalize_ref_list(
    items: list[Any] | None,
    ref_type: str,
    allow_temp: bool = True,
    preserve_invalid: bool = False,
) -> list[Any]:
    normalized: list[Any] = []
    if not isinstance(items, list):
        return normalized

    for item in items:
        ref = None
        if ref_type == "characters":
            ref = normalize_character_ref(item, allow_temp=allow_temp)
        elif ref_type == "environments":
            ref = normalize_environment_ref(item, allow_temp=allow_temp)
        elif ref_type == "plots":
            ref = normalize_plot_ref(item, allow_temp=allow_temp)
        if ref:
            normalized.append(ref)
        elif preserve_invalid:
            normalized.append(item.strip() if isinstance(item, str) else item)
    return normalized


def build_chapter_template(chapter_no: int) -> dict:
    return {
        "chapter_no": chapter_no,
        "pure_plot": "",
        "chapter_goal": "",
        "chapter_conflict": "",
        "chapter_turn": "",
        "chapter_hook_out": "",
        "refs": {
            "characters": [],
            "environments": [],
            "plots": [],
        },
        "expected_deltas": {
            "character_updates": [],
            "environment_updates": [],
            "plot_updates": [],
        },
    }


def _convert_legacy_event(raw_event: dict) -> dict:
    raw_characters = raw_event.get("characters", {})
    raw_environments = raw_event.get("environments", {})
    plots = raw_event.get("plots", [])

    characters = raw_characters if isinstance(raw_characters, dict) else {"existing": raw_characters}
    environments = raw_environments if isinstance(raw_environments, dict) else {"existing": raw_environments}

    char_existing = _normalize_ref_list(characters.get("existing"), "characters", allow_temp=False)
    env_existing = _normalize_ref_list(environments.get("existing"), "environments", allow_temp=False)
    plot_existing = _normalize_ref_list(
        raw_event.get("plot_refs", []),
        "plots",
        allow_temp=False,
        preserve_invalid=True,
    )

    char_new = _normalize_temp_entities(characters.get("new"), CHARACTER_TEMP_PREFIX)
    env_new = _normalize_temp_entities(environments.get("new"), ENVIRONMENT_TEMP_PREFIX)

    chapters: list[dict] = []
    for idx, plot in enumerate(plots, start=1):
        chapter = build_chapter_template(idx)
        if isinstance(plot, dict):
            chapter["pure_plot"] = plot.get("content", "") or plot.get("summary", "")
            chapter["chapter_goal"] = plot.get("title", "")
            chapter["chapter_conflict"] = plot.get("key_conflict", "")
            chapter["chapter_turn"] = plot.get("phase", "")
        chapter["refs"]["characters"] = list(char_existing)
        chapter["refs"]["environments"] = list(env_existing)
        chapter["refs"]["plots"] = list(plot_existing)
        chapters.append(chapter)

    if not chapters:
        chapter = build_chapter_template(1)
        chapter["pure_plot"] = raw_event.get("event", "")
        chapter["refs"]["characters"] = list(char_existing)
        chapter["refs"]["environments"] = list(env_existing)
        chapter["refs"]["plots"] = list(plot_existing)
        chapters.append(chapter)

    return {
        "event_meta": {
            "event_id": raw_event.get("event_id", "event_01"),
            "title": raw_event.get("title", "Untitled Event"),
            "abstract": raw_event.get("event", ""),
            "event_goal": "",
            "core_conflict": "",
            "chapter_count": len(chapters),
            "entry_state": "",
            "exit_hook": "",
        },
        "planning_state": {
            "assumptions": [],
            "open_questions": [],
            "info_sufficiency": {
                "status": "complete",
                "missing_must_have": [],
            },
        },
        "seed_subgraph": {
            "characters": {
                "existing": char_existing,
                "new": char_new,
                "relationships": [],
            },
            "environments": {
                "existing": env_existing,
                "new": env_new,
            },
            "plots": {
                "existing": plot_existing,
                "unresolved_threads": [],
                "new": [],
            },
        },
        "chapters": chapters,
    }


def normalize_event_payload(raw_event: dict | None) -> dict | None:
    if not isinstance(raw_event, dict):
        return None

    if "event_meta" not in raw_event or "chapters" not in raw_event:
        normalized = _convert_legacy_event(raw_event)
    else:
        normalized = copy.deepcopy(raw_event)

    normalized.setdefault("planning_state", {})
    normalized["planning_state"].setdefault("assumptions", [])
    normalized["planning_state"].setdefault("open_questions", [])
    normalized["planning_state"].setdefault("info_sufficiency", {})
    normalized["planning_state"]["info_sufficiency"].setdefault("status", "complete")
    normalized["planning_state"]["info_sufficiency"].setdefault("missing_must_have", [])

    normalized.setdefault("seed_subgraph", {})
    normalized["seed_subgraph"].setdefault("characters", {"existing": [], "new": []})
    normalized["seed_subgraph"].setdefault("environments", {"existing": [], "new": []})
    normalized["seed_subgraph"].setdefault("plots", {"existing": [], "unresolved_threads": [], "new": []})

    normalized["seed_subgraph"]["characters"]["existing"] = _normalize_ref_list(
        normalized["seed_subgraph"]["characters"].get("existing"),
        "characters",
        allow_temp=False,
    )
    normalized["seed_subgraph"]["environments"]["existing"] = _normalize_ref_list(
        normalized["seed_subgraph"]["environments"].get("existing"),
        "environments",
        allow_temp=False,
    )
    normalized["seed_subgraph"]["plots"]["existing"] = _normalize_ref_list(
        normalized["seed_subgraph"]["plots"].get("existing"),
        "plots",
        allow_temp=False,
        preserve_invalid=True,
    )
    normalized["seed_subgraph"]["plots"]["unresolved_threads"] = _normalize_ref_list(
        normalized["seed_subgraph"]["plots"].get("unresolved_threads"),
        "plots",
        allow_temp=False,
        preserve_invalid=True,
    )
    normalized["seed_subgraph"]["characters"]["new"] = _normalize_temp_entities(
        normalized["seed_subgraph"]["characters"].get("new"),
        CHARACTER_TEMP_PREFIX,
    )
    raw_relationships = normalized["seed_subgraph"]["characters"].get("relationships", [])
    normalized["seed_subgraph"]["characters"]["relationships"] = copy.deepcopy(raw_relationships)
    normalized["seed_subgraph"]["environments"]["new"] = _normalize_temp_entities(
        normalized["seed_subgraph"]["environments"].get("new"),
        ENVIRONMENT_TEMP_PREFIX,
    )
    legacy_plot_new = normalized["seed_subgraph"]["plots"].get("new", [])
    normalized["seed_subgraph"]["plots"]["new"] = copy.deepcopy(legacy_plot_new)

    chapters = normalized.get("chapters", [])
    if not isinstance(chapters, list):
        chapters = []

    final_chapters: list[dict] = []
    for index, chapter in enumerate(chapters, start=1):
        merged = build_chapter_template(index)
        if isinstance(chapter, dict):
            merged.update(chapter)
            refs = chapter.get("refs", {})
            merged["refs"] = {
                "characters": _normalize_ref_list(refs.get("characters"), "characters"),
                "environments": _normalize_ref_list(refs.get("environments"), "environments"),
                "plots": _normalize_ref_list(
                    refs.get("plots"),
                    "plots",
                    allow_temp=False,
                    preserve_invalid=True,
                ),
            }
            expected_deltas = chapter.get("expected_deltas", {})
            merged["expected_deltas"] = {
                "character_updates": list(expected_deltas.get("character_updates", [])),
                "environment_updates": list(expected_deltas.get("environment_updates", [])),
                "plot_updates": list(expected_deltas.get("plot_updates", [])),
            }
        merged["chapter_no"] = index
        final_chapters.append(merged)
    normalized["chapters"] = final_chapters

    normalized.setdefault("event_meta", {})
    normalized["event_meta"].setdefault("event_id", "event_01")
    normalized["event_meta"].setdefault("title", "Untitled Event")
    normalized["event_meta"].setdefault("abstract", "")
    normalized["event_meta"].setdefault("event_goal", "")
    normalized["event_meta"].setdefault("core_conflict", "")
    normalized["event_meta"].setdefault("entry_state", "")
    normalized["event_meta"].setdefault("exit_hook", "")
    normalized["event_meta"]["chapter_count"] = len(final_chapters)
    return normalized


def build_ref_registry(event_payload: dict) -> dict[str, set[str]]:
    seed_subgraph = event_payload.get("seed_subgraph", {})
    plots_section = seed_subgraph.get("plots", {})
    if not isinstance(plots_section, dict):
        plots_section = {}
    plot_existing = plots_section.get("existing", [])
    plot_unresolved = plots_section.get("unresolved_threads", [])
    if not isinstance(plot_existing, list):
        plot_existing = []
    if not isinstance(plot_unresolved, list):
        plot_unresolved = []
    character_new = {
        item["temp_id"]
        for item in seed_subgraph.get("characters", {}).get("new", [])
        if isinstance(item, dict) and _is_non_empty_text(item.get("temp_id"))
    }
    environment_new = {
        item["temp_id"]
        for item in seed_subgraph.get("environments", {}).get("new", [])
        if isinstance(item, dict) and _is_non_empty_text(item.get("temp_id"))
    }
    return {
        "characters": set(seed_subgraph.get("characters", {}).get("existing", [])) | character_new,
        "environments": set(seed_subgraph.get("environments", {}).get("existing", [])) | environment_new,
        "plots": {
            ref
            for ref in (plot_existing + plot_unresolved)
            if is_real_plot_ref(ref)
        },
    }


def validate_event_payload(event_payload: dict | None) -> list[str]:
    if not isinstance(event_payload, dict):
        return ["Event payload must be a JSON object."]

    errors: list[str] = []
    event_meta = event_payload.get("event_meta")
    if not isinstance(event_meta, dict):
        return ["Missing event_meta object."]

    chapters = event_payload.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        return ["Event must contain at least 1 chapter."]

    if event_meta.get("chapter_count") != len(chapters):
        errors.append("event_meta.chapter_count must equal the number of chapters.")
    if len(chapters) != 2:
        errors.append("Under the current paper experiment configuration, each Event must contain exactly 2 chapters.")

    planning_state = event_payload.get("planning_state", {})
    info_sufficiency = planning_state.get("info_sufficiency", {})
    missing_must_have = info_sufficiency.get("missing_must_have", [])
    if missing_must_have:
        errors.append("planning_state.info_sufficiency.missing_must_have must be empty before reaching the final Event.")

    registry = build_ref_registry(event_payload)

    character_seed = event_payload.get("seed_subgraph", {}).get("characters", {})
    proposed_relationships = (
        character_seed.get("relationships", []) if isinstance(character_seed, dict) else []
    )
    if not isinstance(proposed_relationships, list):
        errors.append("seed_subgraph.characters.relationships must be an array.")
    else:
        seen_relationships: set[tuple[str, str]] = set()
        for index, relationship in enumerate(proposed_relationships):
            if not isinstance(relationship, dict):
                errors.append(f"characters.relationships[{index}] must be an object.")
                continue
            source = relationship.get("source")
            target = relationship.get("target")
            endpoints_valid = not (
                not isinstance(source, str)
                or not isinstance(target, str)
                or source not in registry["characters"]
                or target not in registry["characters"]
            )
            if not endpoints_valid:
                errors.append(
                    f"characters.relationships[{index}] endpoints must be registered character refs."
                )
            elif source == target:
                errors.append(f"characters.relationships[{index}] may not be a self-edge.")
            if not _is_non_empty_text(relationship.get("current_type")):
                errors.append(f"characters.relationships[{index}] is missing current_type.")
            change_history = relationship.get("change_history", [])
            if not isinstance(change_history, list):
                errors.append(f"characters.relationships[{index}].change_history must be an array.")
            else:
                if not change_history:
                    errors.append(
                        f"characters.relationships[{index}].change_history must contain the initial H_ij transition."
                    )
                for history_index, history_entry in enumerate(change_history):
                    history_path = f"characters.relationships[{index}].change_history[{history_index}]"
                    if not isinstance(history_entry, dict):
                        errors.append(f"{history_path} must be an object.")
                        continue
                    for field_name in ("from", "to", "reason"):
                        if not _is_non_empty_text(history_entry.get(field_name)):
                            errors.append(f"{history_path}.{field_name} must be non-empty text.")
            if endpoints_valid:
                edge_key = (source, target)
                if edge_key in seen_relationships:
                    errors.append(f"characters.relationships[{index}] duplicates an earlier edge.")
                seen_relationships.add(edge_key)

    seed_subgraph = event_payload.get("seed_subgraph", {})
    plots_seed = seed_subgraph.get("plots", {}) if isinstance(seed_subgraph, dict) else {}
    if not isinstance(plots_seed, dict):
        errors.append("seed_subgraph.plots must be an object.")
        plots_seed = {}
    planned_plots = plots_seed.get("new", [])
    if not isinstance(planned_plots, list):
        errors.append("seed_subgraph.plots.new must be an empty array.")
    elif planned_plots:
        errors.append(
            "seed_subgraph.plots.new must be empty; new plot nodes are created only by reverse canon extraction."
        )
    for field_name in ("existing", "unresolved_threads"):
        refs = plots_seed.get(field_name, [])
        if not isinstance(refs, list):
            errors.append(f"seed_subgraph.plots.{field_name} must be an array.")
            continue
        for ref in refs:
            if not is_real_plot_ref(ref):
                errors.append(
                    f"seed_subgraph.plots.{field_name} may only contain existing p_K refs: {ref}"
                )
    existing_plot_refs = {
        ref for ref in plots_seed.get("existing", [])
        if is_real_plot_ref(ref)
    } if isinstance(plots_seed.get("existing", []), list) else set()
    unresolved_plot_refs = plots_seed.get("unresolved_threads", [])
    if isinstance(unresolved_plot_refs, list):
        for ref in unresolved_plot_refs:
            if is_real_plot_ref(ref) and ref not in existing_plot_refs:
                errors.append(
                    f"seed_subgraph.plots.unresolved_threads ref must also be registered in plots.existing: {ref}"
                )

    for index, chapter in enumerate(chapters, start=1):
        if not isinstance(chapter, dict):
            errors.append(f"Chapter {index} must be an object.")
            continue
        if chapter.get("chapter_no") != index:
            errors.append(f"Chapter {index} chapter_no must increment sequentially.")
        if not _is_non_empty_text(chapter.get("pure_plot")):
            errors.append(f"Chapter {index} is missing pure_plot.")

        refs = chapter.get("refs")
        if not isinstance(refs, dict):
            errors.append(f"Chapter {index} is missing refs.")
            continue

        if not refs.get("characters") or not refs.get("environments"):
            errors.append(f"Chapter {index} must contain both character and environment refs.")

        for ref_type, checker in (
            ("characters", is_character_ref),
            ("environments", is_environment_ref),
            # Plot refs are historical canon only. New plots are extracted from
            # chapter prose after writing and therefore have no Event-stage ID.
            ("plots", is_real_plot_ref),
        ):
            ref_items = refs.get(ref_type)
            if not isinstance(ref_items, list):
                errors.append(f"Chapter {index} refs.{ref_type} must be an array.")
                continue
            for ref in ref_items:
                if not isinstance(ref, str):
                    errors.append(f"Chapter {index} refs.{ref_type} may only contain string IDs.")
                    continue
                if not checker(ref):
                    errors.append(f"Chapter {index} refs.{ref_type} contains an illegal ID: {ref}")
                    continue
                if ref not in registry[ref_type]:
                    errors.append(f"Chapter {index} refs.{ref_type} contains an unregistered reference: {ref}")

        expected_deltas = chapter.get("expected_deltas")
        if not isinstance(expected_deltas, dict):
            errors.append(f"Chapter {index} is missing expected_deltas.")
            continue
        for key in ("character_updates", "environment_updates", "plot_updates"):
            if not isinstance(expected_deltas.get(key, []), list):
                errors.append(f"Chapter {index} expected_deltas.{key} must be an array.")

    return errors


def get_chapter_payload(event_payload: dict, chapter_no: int) -> dict | None:
    chapters = event_payload.get("chapters", [])
    for chapter in chapters:
        if isinstance(chapter, dict) and chapter.get("chapter_no") == chapter_no:
            return chapter
    return None


def dump_event_payload(event_payload: dict) -> str:
    return json.dumps(event_payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Active Reference Set (R_active)
#
#   R_active = { R_char, R_env, R_plot }
# ---------------------------------------------------------------------------


def extract_active_reference_set(event_payload: dict) -> dict[str, list[str]]:
    if not isinstance(event_payload, dict):
        return {"characters": [], "environments": [], "plots": []}

    seed = event_payload.get("seed_subgraph", {}) or {}
    chapters = event_payload.get("chapters", []) or []

    def _ordered_unique(values):
        seen: set[str] = set()
        out: list[str] = []
        for v in values:
            if isinstance(v, str) and v and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    char_refs = list(seed.get("characters", {}).get("existing", []) or [])
    env_refs = list(seed.get("environments", {}).get("existing", []) or [])
    char_refs += [
        item.get("temp_id")
        for item in seed.get("characters", {}).get("new", []) or []
        if isinstance(item, dict)
    ]
    env_refs += [
        item.get("temp_id")
        for item in seed.get("environments", {}).get("new", []) or []
        if isinstance(item, dict)
    ]
    plot_refs = list(seed.get("plots", {}).get("existing", []) or [])
    plot_refs += list(seed.get("plots", {}).get("unresolved_threads", []) or [])

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        refs = chapter.get("refs", {}) or {}
        char_refs += refs.get("characters", []) or []
        env_refs += refs.get("environments", []) or []
        plot_refs += refs.get("plots", []) or []

    plot_refs = [ref for ref in plot_refs if is_real_plot_ref(ref)]

    return {
        "characters": _ordered_unique(char_refs),
        "environments": _ordered_unique(env_refs),
        "plots": _ordered_unique(plot_refs),
    }


def attach_active_reference_set(event_payload: dict) -> dict:
    if not isinstance(event_payload, dict):
        return event_payload
    event_payload["R_active"] = extract_active_reference_set(event_payload)
    return event_payload
