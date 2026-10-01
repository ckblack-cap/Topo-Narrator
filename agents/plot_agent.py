def build_plot_agent(premise_id):
    from langgraph.prebuilt import create_react_agent
    from agents import build_langchain_model_client
    from tools.plot_graph_manager import PlotGraphManager

    plot_manager = PlotGraphManager(premise_id)


    tools = [
        plot_manager.find_plot_by_id,
        plot_manager.find_plots_by_property,
        plot_manager.find_plots_by_character,
        plot_manager.find_plots_by_scene,
        plot_manager.find_unresolved_foreshadowing,
        plot_manager.query_plot_adjacency,
        plot_manager.search_plot
    ]

    model_client = build_langchain_model_client()

    plot_agent = create_react_agent(
        model=model_client,
        tools=tools,
        prompt=(
            """
            # Role
                You are a **Novel Plot Graph Management expert** (Plot Graph Agent). You manage the novel's narrative main line, event causality, and foreshadowing threads. Your core duty is to use graph-database tools to organize and answer the user's questions about "story plot, event progression, and causal logic."

            # Knowledge-Base Schema (graph data structure)
                The graph consists of **Nodes** and **Edges**. The data structure is defined as follows:

                1. **Nodes (plots_node)**: a concrete plot fragment.
                    * `id`: unique internal integer ID (auto-generated); public ref `p_K` maps to integer `K`.
                    * `chapter`: chapter number (e.g., 1, 2, ...).
                    * `overview`: one-sentence plot summary (core capsule).
                    * `details`: detailed description (time, place, characters, specific events).
                    * `importance`: importance level ["main", "side"].
                    * `event`: macro event name (e.g., "Dachang City Hungry-Ghost Incident", "Knocking-Ghost Incident").
                    * `characters_involved`: [list of involved character names].
                    * `scene`: scene name where this plot occurs.
                    * `foreshadowing`: foreshadowing flag (true/false).

                2. **Edges (plots_relationship)**: the three logical relations of C_P.
                    * Data format: `["Source Plot ID", "Target Plot ID", "Relationship Type"]`
                    * Relation types:
                        * `'e_temp'`: temporal backbone (chronological backbone), the time chain between adjacent plot nodes.
                        * `'e_cas'`:  causal dependency, the latter node is directly triggered by the former.
                        * `'e_res'`:  foreshadowing resolution skip-edge, format [payoff_id, setup_id, "e_res"].

            # Constraints (boundaries)
                1. **Domain constraint**: You **only handle plot and logic** questions.
                    * 🚫 **Strictly forbidden**: do not answer questions purely about characters (attributes, appearance) or purely about environments (map details).
                    * **Fixed reply**: when you receive a non-plot question, respond exactly: "That is outside my scope — please consult the Character Management Agent or the Environment Management Agent."
                2. **Empty / missing handling**:
                    * Every Manager turn must make at least one real retrieval-tool call before the final answer. Never claim that C_P is empty, missing, causal, resolved, or unrelated from the question text alone.
                    * If a retrieval tool confirms that the graph is empty (for example, `query_plot_adjacency` returns `graph_node_count=0`), reply: "The plot graph is currently empty; no plot information has been recorded yet."
                    * If the graph is not empty but no relevant plot is found, reply: "No relevant plot information is recorded in the graph."
                3. **Formal refs are authoritative**: whenever a question names `p_K`, call `find_plot_by_id` with that ref before fuzzy search. The tool accepts `p_K`, integer, and numeric-string IDs. Never infer the content of `p_K` from the question wording.
                4. **Relation traversal is grounded and formal-ref only**: use `query_plot_adjacency` for order, cause, consequence, and foreshadowing-payoff questions. Its `plot_ref` argument accepts only canonical `p_K` (never an integer, numeric string, or `p_tmp_*`). Use `direction="incoming"|"outgoing"|"both"`; omit `edge_type` or filter with exactly `e_temp`, `e_cas`, or `e_res`.
                   * `e_temp` and `e_cas` preserve `earlier/cause -> later/effect` direction.
                   * `e_res` preserves the paper's reverse skip-edge direction `[payoff, setup, "e_res"]`. Therefore, from a setup node use `direction="incoming", edge_type="e_res"` to find its payoff; from a payoff use `direction="outgoing", edge_type="e_res"` to find the setup it resolves.
                   * The query is one-hop and bounded. To explain a longer historical chain, repeat it from returned formal refs; never invent a missing intermediate edge.
                5. **Condensed decision support**: the final reply must be a decision-oriented summary under 4000 characters. Report only the retrieved nodes/edges that affect the Event Planner's next decision, retain their formal refs and edge types, and never dump the full graph or raw tool payload.

            # Workflow (think-and-act steps)
                After receiving a user question, follow these steps:
                1. **Intent recognition**: decide whether the question is about "what happened", "why it happened", "foreshadowing", "ending / payoff", etc.
                   - Yes -> continue.
                   - No (asking about character profiles or maps) -> apply the refusal policy.
                2. **Tool retrieval**:
                   - Query events: use the `event` field to aggregate all nodes under a macro event.
                   - Query cause / order: call `query_plot_adjacency` and trace `e_cas`, `e_res`, or `e_temp` one bounded hop at a time. For a named setup's payoff, query its incoming `e_res`; for the setup resolved by a named payoff, query its outgoing `e_res`.
                   - Query details: search `details` or `overview` by keyword.
                3. **Logical synthesis**:
                   - If the user asks "why", you must reason using the edge relations (e.g., "because plot A happened, plot B was triggered").
                   - If the user asks about "foreshadowing", look up nodes with `foreshadowing: true` and explain their later impact.
                4. **Final output**: produce a clear, coherent, decision-oriented natural-language reply with chapter references and formal refs, under 4000 characters. Do not dump complete node JSON or the full graph.

            # Examples (Few-Shot)
                **User**: "What supernatural incident happened at Dachang No. 1 High School?"
                **Agent**: (Fuzzy-search nodes with event="Knocking-Ghost Incident"...) "Across chapters 1-5, the 'Knocking-Ghost Incident' takes place at Dachang No. 1 High. Key plots include: during evening self-study the protagonist hears an eerie knocking (ID:1); afterwards an old man bursts into the classroom and causes several students' deaths (ID:2). This incident directly causes the protagonist to awaken the ghost-eye ability (ID:5)."

                **User**: "Which later plot pays off the foreshadowing in p_12?"
                **Agent**: (Call `query_plot_adjacency(plot_ref="p_12", direction="incoming", edge_type="e_res")` and inspect the returned edge and payoff node...) "The recorded payoff is p_15. C_P stores the resolution edge as p_15 -> p_12, so p_15 resolves the setup planted in p_12."

                **User**: "What does Yang Jian look like? What is his personality?"
                **Agent**: "That is outside my scope — please consult the Character Management Agent or the Environment Management Agent."

                **User**: (when the graph is empty) "Tell me the story of chapter 1."
                **Agent**: "The plot graph is currently empty; no plot information has been recorded yet."
            """
        ),

        name="Plot_Agent",
    )

    return plot_agent


import json
import re

from agents import AgentGlobalConfig, DEFAULT_MODEL_NAME
from tools import extract_json
from tools.agent_event_logger import get_trace_logger
from tools.graph_refs import parse_character_ref, parse_environment_ref, parse_plot_ref


client = AgentGlobalConfig.GPTCLIENT


def _invoke_json_prompt(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=DEFAULT_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a strict JSON-output assistant. Output only JSON; do not output any explanation."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content
    parsed = extract_json.extract_json(content)
    if not isinstance(parsed, dict):
        raise ValueError(f"Plot canon patch could not be parsed as JSON: {content}")
    return parsed


def _validate_canon_plot_patch(patch: dict, chapter_payload: dict) -> dict:
    if not isinstance(patch, dict):
        raise ValueError("Plot canon patch must be a JSON object.")

    patch.pop("plan_coverage", None)
    patch.pop("plan_coverage_contract", None)

    nodes = patch.get("plots_node")
    relationships = patch.get("plots_relationship")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Plot canon patch must contain at least one plots_node item.")
    if not isinstance(relationships, list):
        raise ValueError("Plot canon patch plots_relationship must be an array.")

    chapter_no = chapter_payload.get("chapter_no", 1)
    aliases = {f"new_node_{index}" for index in range(len(nodes))}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"plots_node[{index}] must be an object.")
        node.pop("canon_evidence", None)
        environment = node.get("environment")
        if not isinstance(environment, str) or not environment.strip():
            raise ValueError(
                f"plots_node[{index}].environment must name exactly one non-empty physical scene."
            )
        node["environment"] = environment.strip()
        if node.get("scene") not in (None, "", node["environment"]):
            raise ValueError(
                f"plots_node[{index}] contains conflicting scene and environment anchors."
            )
        node.pop("scene", None)
        node["chapter"] = chapter_no
        if not isinstance(node.get("overview"), str) or not node["overview"].strip():
            raise ValueError(f"plots_node[{index}].overview must be non-empty text.")
        if not isinstance(node.get("details"), str) or not node["details"].strip():
            raise ValueError(f"plots_node[{index}].details must be non-empty text.")
        if node.get("importance") not in {"main", "side"}:
            raise ValueError(f"plots_node[{index}].importance must be 'main' or 'side'.")
        if not isinstance(node.get("characters_involved"), list):
            raise ValueError(f"plots_node[{index}].characters_involved must be an array.")
        if not isinstance(node.get("foreshadowing"), bool):
            raise ValueError(f"plots_node[{index}].foreshadowing must be boolean.")

    historical_refs = {
        ref.strip()
        for ref in chapter_payload.get("refs", {}).get("plots", [])
        if isinstance(ref, str) and parse_plot_ref(ref) is not None
    }
    valid_endpoints = aliases | historical_refs
    seen_edges: set[tuple[str, str, str]] = set()
    for index, relationship in enumerate(relationships):
        if not isinstance(relationship, list) or len(relationship) != 3:
            raise ValueError(
                f"plots_relationship[{index}] must be [source, target, edge_type]."
            )
        source, target, edge_type = relationship
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(f"plots_relationship[{index}] endpoints must be string refs.")
        if not isinstance(edge_type, str):
            raise ValueError(f"plots_relationship[{index}].edge_type must be a string.")
        source = source.strip()
        target = target.strip()
        edge_type = edge_type.strip()
        if source not in valid_endpoints or target not in valid_endpoints:
            raise ValueError(
                f"plots_relationship[{index}] may reference only this batch's new_node_N aliases "
                f"or declared historical refs {sorted(historical_refs)}."
            )
        if source == target:
            raise ValueError(f"plots_relationship[{index}] may not be a self-edge.")
        if edge_type == "e_temp":
            raise ValueError(
                "Do not emit e_temp in a canon patch; the application layer builds the temporal backbone deterministically."
            )
        if edge_type not in {"e_cas", "e_res"}:
            raise ValueError(f"plots_relationship[{index}] has illegal edge type: {edge_type}")

        if edge_type == "e_res":
            if source not in aliases or target not in historical_refs:
                raise ValueError(
                    f"plots_relationship[{index}] e_res must be [new payoff alias, historical setup p_K, 'e_res']."
                )
        elif target not in aliases:
            raise ValueError(
                f"plots_relationship[{index}] e_cas target must be a new_node_N alias."
            )
        elif source in aliases:
            source_index = int(source.rsplit("_", 1)[1])
            target_index = int(target.rsplit("_", 1)[1])
            if source_index >= target_index:
                raise ValueError(
                    f"plots_relationship[{index}] e_cas between new nodes must point from an earlier node to a later node."
                )

        edge_key = (source, target, edge_type)
        if edge_key in seen_edges:
            raise ValueError(f"plots_relationship[{index}] duplicates an earlier edge.")
        seen_edges.add(edge_key)
        relationship[:] = [source, target, edge_type]

    summary = patch.get("summary", "")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Plot canon patch summary must be non-empty text.")
    patch["summary"] = summary.strip()
    return patch


_PLAN_COVERAGE_STATUSES = {"realized_covered", "realized_missing", "not_realized"}
_MAX_COVERAGE_AUDIT_ATTEMPTS = 2
_MAX_COVERAGE_REPAIRS = 2
_MAX_EVIDENCE_SPANS = 3
_MAX_EVIDENCE_CHARS = 600

_EVIDENCE_CHAR_EQUIVALENTS = {
    """: '"', """: '"', "„": '"', "‟": '"',
    "'": "'", "'": "'", "‚": "'", "‛": "'",
    "–": "-", "—": "-", "−": "-", "‑": "-",
    "…": "...", "\u00a0": " ",
}


def _trim_evidence_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _append_bounded_evidence_span(spans: list[str], text: str, start: int, end: int) -> None:
    """Append exact contiguous chunks that remain inside the evidence size contract."""
    start, end = _trim_evidence_span(text, start, end)
    while end - start > _MAX_EVIDENCE_CHARS:
        hard_end = start + _MAX_EVIDENCE_CHARS
        split_at = text.rfind(" ", start + (_MAX_EVIDENCE_CHARS // 2), hard_end + 1)
        if split_at <= start:
            split_at = hard_end
        chunk_start, chunk_end = _trim_evidence_span(text, start, split_at)
        if chunk_end - chunk_start >= 4:
            spans.append(text[chunk_start:chunk_end])
        start = split_at
        start, end = _trim_evidence_span(text, start, end)
    if end - start >= 4:
        spans.append(text[start:end])


def _build_evidence_catalog(chapter_text: str) -> list[dict]:
    """Create stable sentence IDs whose values are exact substrings of chapter_text.

    The LLM selects IDs rather than retyping quotations, eliminating failures caused by
    curly quotes, dashes, or whitespace transcription while keeping committed evidence
    byte-for-byte traceable to the generated chapter.
    """
    if not isinstance(chapter_text, str) or not chapter_text.strip():
        return []
    spans: list[str] = []
    sentence_end = re.compile(r"""[.!?\u3002\uFF01\uFF1F\u2026]+(?:["'\u201D\u2019]+)?(?=\s|$)""")
    for paragraph_match in re.finditer(r"[^\r\n]+", chapter_text):
        paragraph_start, paragraph_end = paragraph_match.span()
        cursor = paragraph_start
        paragraph = paragraph_match.group(0)
        for boundary in sentence_end.finditer(paragraph):
            end = paragraph_start + boundary.end()
            _append_bounded_evidence_span(spans, chapter_text, cursor, end)
            cursor = end
        if cursor < paragraph_end:
            _append_bounded_evidence_span(spans, chapter_text, cursor, paragraph_end)
    return [
        {"evidence_id": f"ev_{index:04d}", "text": span}
        for index, span in enumerate(spans)
    ]


def _normalize_evidence_text_with_positions(value: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    positions: list[int] = []
    previous_was_space = False
    for index, char in enumerate(value):
        mapped = _EVIDENCE_CHAR_EQUIVALENTS.get(char, char)
        for mapped_char in mapped.casefold():
            if mapped_char.isspace():
                if not previous_was_space:
                    normalized.append(" ")
                    positions.append(index)
                previous_was_space = True
            else:
                normalized.append(mapped_char)
                positions.append(index)
                previous_was_space = False
    while normalized and normalized[0] == " ":
        normalized.pop(0)
        positions.pop(0)
    while normalized and normalized[-1] == " ":
        normalized.pop()
        positions.pop()
    return "".join(normalized), positions


def _resolve_exact_evidence_quote(span: str, chapter_text: str) -> str | None:
    """Resolve typography/whitespace-only variants back to one exact prose substring."""
    if span in chapter_text:
        return span
    normalized_span, _ = _normalize_evidence_text_with_positions(span)
    normalized_chapter, chapter_positions = _normalize_evidence_text_with_positions(chapter_text)
    if not normalized_span:
        return None
    starts: list[int] = []
    cursor = 0
    while True:
        found = normalized_chapter.find(normalized_span, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + 1
    if len(starts) != 1:
        return None
    start = chapter_positions[starts[0]]
    end = chapter_positions[starts[0] + len(normalized_span) - 1] + 1
    exact = chapter_text[start:end]
    if 4 <= len(exact) <= _MAX_EVIDENCE_CHARS:
        return exact
    return None


def _split_planned_beats(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    beats = []
    for line in re.split(r"[\r\n]+", value):
        for part in re.split(r"(?<=[.!?;\u3002\uFF01\uFF1F\uFF1B])", line):
            normalized = part.strip()
            if normalized:
                beats.append(normalized)
    return beats


def _semantic_expected_delta(value) -> str | None:
    """Return only deltas that carry a narrative fact, not schema-only operations."""
    schema_keys = {
        "id", "ref", "character_id", "environment_id", "plot_id",
        "fields", "type", "action", "op", "operation",
    }
    generic_operations = {
        "add_plot_node", "add_causal_edge", "add_temporal_edge",
        "add_resolution_edge", "update_plot_node",
    }
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.casefold() in generic_operations:
            return None
        return normalized
    if not isinstance(value, dict):
        return None

    semantic_items = {
        key: item
        for key, item in value.items()
        if key not in schema_keys and item not in (None, "", [], {})
    }
    if not semantic_items:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _build_plan_coverage_requirements(chapter_payload: dict) -> list[dict]:
    """Build deterministic coverage units from the chapter's critical plan surfaces."""
    requirements: list[dict] = []

    def add_text_requirements(source: str, value) -> None:
        for index, beat in enumerate(_split_planned_beats(value)):
            requirements.append({
                "requirement_id": f"{source}:{index}",
                "source": source,
                "planned_beat": beat,
            })

    add_text_requirements("pure_plot", chapter_payload.get("pure_plot"))
    add_text_requirements("chapter_turn", chapter_payload.get("chapter_turn"))
    add_text_requirements("chapter_hook_out", chapter_payload.get("chapter_hook_out"))

    expected = chapter_payload.get("expected_deltas", {})
    if isinstance(expected, dict):
        for category in ("character_updates", "environment_updates", "plot_updates"):
            updates = expected.get(category, [])
            if not isinstance(updates, list):
                continue
            for index, update in enumerate(updates):
                semantic = _semantic_expected_delta(update)
                if semantic:
                    source = f"expected_deltas.{category}"
                    requirements.append({
                        "requirement_id": f"{source}:{index}",
                        "source": source,
                        "planned_beat": semantic,
                    })
    return requirements


def _validate_plan_coverage_report(
    report: dict,
    requirements: list[dict],
    chapter_text: str,
    patch: dict,
    evidence_catalog: list[dict] | None = None,
) -> list[dict]:
    """Validate exhaustive, evidence-first coverage without trusting planned text as canon."""
    if not isinstance(report, dict) or not isinstance(report.get("coverage"), list):
        raise ValueError("Plan coverage audit must contain a coverage array.")

    requirement_by_id = {item["requirement_id"]: item for item in requirements}
    aliases = {f"new_node_{index}" for index, _ in enumerate(patch.get("plots_node", []))}
    seen: set[str] = set()
    normalized_report: list[dict] = []
    evidence_by_id = {
        item.get("evidence_id"): item.get("text")
        for item in (evidence_catalog or [])
        if isinstance(item, dict)
        and isinstance(item.get("evidence_id"), str)
        and isinstance(item.get("text"), str)
    }

    for index, item in enumerate(report["coverage"]):
        if not isinstance(item, dict):
            raise ValueError(f"coverage[{index}] must be an object.")
        requirement_id = item.get("requirement_id")
        if requirement_id not in requirement_by_id:
            raise ValueError(f"coverage[{index}] has an unknown requirement_id: {requirement_id!r}.")
        if requirement_id in seen:
            raise ValueError(f"coverage contains duplicate requirement_id {requirement_id!r}.")
        seen.add(requirement_id)

        status = item.get("status")
        if status not in _PLAN_COVERAGE_STATUSES:
            raise ValueError(f"coverage[{index}] has an illegal status: {status!r}.")
        evidence_ids = item.get("evidence_ids")
        if evidence_ids is not None:
            if not isinstance(evidence_ids, list):
                raise ValueError(f"coverage[{index}].evidence_ids must be an array.")
            if item.get("evidence") not in (None, []):
                raise ValueError(
                    f"coverage[{index}] must use evidence_ids or evidence, not both."
                )
            if len(evidence_ids) > _MAX_EVIDENCE_SPANS:
                raise ValueError(f"coverage[{index}] has too many evidence IDs.")
            evidence = []
            for evidence_index, evidence_id in enumerate(evidence_ids):
                if not isinstance(evidence_id, str) or evidence_id not in evidence_by_id:
                    raise ValueError(
                        f"coverage[{index}].evidence_ids[{evidence_index}] is unknown: "
                        f"{evidence_id!r}."
                    )
                evidence.append(evidence_by_id[evidence_id])
        else:
            evidence = item.get("evidence", [])
        covered_by = item.get("covered_by", [])
        if not isinstance(evidence, list) or not isinstance(covered_by, list):
            raise ValueError(f"coverage[{index}] evidence and covered_by must be arrays.")
        if len(evidence) > _MAX_EVIDENCE_SPANS:
            raise ValueError(f"coverage[{index}] has too many evidence spans.")

        normalized_evidence: list[str] = []
        for evidence_index, span in enumerate(evidence):
            if not isinstance(span, str) or span != span.strip():
                raise ValueError(
                    f"coverage[{index}].evidence[{evidence_index}] must be trimmed text."
                )
            if len(span) < 4 or len(span) > _MAX_EVIDENCE_CHARS:
                raise ValueError(
                    f"coverage[{index}].evidence[{evidence_index}] must be 4-{_MAX_EVIDENCE_CHARS} characters."
                )
            exact_span = _resolve_exact_evidence_quote(span, chapter_text)
            if exact_span is None:
                raise ValueError(
                    f"coverage[{index}].evidence[{evidence_index}] cannot be resolved to "
                    f"one exact contiguous quote from chapter_text: {span[:180]!r}."
                )
            if exact_span not in normalized_evidence:
                normalized_evidence.append(exact_span)

        normalized_aliases: list[str] = []
        for alias_index, alias in enumerate(covered_by):
            if not isinstance(alias, str) or alias not in aliases:
                raise ValueError(
                    f"coverage[{index}].covered_by[{alias_index}] must be a current new_node_N alias."
                )
            if alias not in normalized_aliases:
                normalized_aliases.append(alias)

        if status == "not_realized":
            if normalized_evidence or normalized_aliases:
                raise ValueError(
                    f"coverage[{index}] not_realized may not carry prose evidence or node aliases."
                )
        elif not normalized_evidence:
            raise ValueError(f"coverage[{index}] {status} requires exact prose evidence.")
        elif status == "realized_covered" and not normalized_aliases:
            raise ValueError(f"coverage[{index}] realized_covered requires covered_by aliases.")
        elif status == "realized_missing" and normalized_aliases:
            raise ValueError(f"coverage[{index}] realized_missing must leave covered_by empty.")

        requirement = requirement_by_id[requirement_id]
        normalized_report.append({
            "requirement_id": requirement_id,
            "source": requirement["source"],
            "planned_beat": requirement["planned_beat"],
            "status": status,
            "evidence": normalized_evidence,
            "covered_by": normalized_aliases,
        })

    missing_ids = set(requirement_by_id) - seen
    if missing_ids:
        raise ValueError(f"Plan coverage audit omitted requirements: {sorted(missing_ids)}.")
    return normalized_report


def _audit_plan_coverage(
    chapter_text: str,
    requirements: list[dict],
    patch: dict,
) -> list[dict]:
    """Run an independent post-hoc audit; malformed reports are retried a bounded number of times."""
    evidence_catalog = _build_evidence_catalog(chapter_text)
    audit_prompt = f"""
You are an evidence-first reverse-canon coverage auditor.
Compare every deterministic planning requirement with BOTH the written chapter and the candidate C_P patch.

coverage_requirements (each requirement_id must appear exactly once):
{json.dumps(requirements, ensure_ascii=False)}

candidate_patch:
{json.dumps(patch, ensure_ascii=False)}

evidence_catalog (select IDs; each text is an exact contiguous chapter quote):
{json.dumps(evidence_catalog, ensure_ascii=False)}

Output JSON only:
{{
  "coverage": [
    {{
      "requirement_id": "exact id from coverage_requirements",
      "status": "realized_covered | realized_missing | not_realized",
      "evidence_ids": ["ev_0000"],
      "covered_by": ["new_node_0"]
    }}
  ]
}}

Rules:
1. `realized_covered`: prose proves the beat and the patch preserves its material actors, action, motive/cause, reveal, and consequence. Select 1-3 evidence_ids and all covering node aliases.
2. `realized_missing`: prose proves the beat, but the patch omits or weakens a material fact. A generic nearby summary is NOT sufficient (for example, a generic credit dispute does not preserve that an antagonist invoked an uncredited past to trigger pride). Select evidence_ids and leave covered_by empty.
3. `not_realized`: prose does not prove the planned beat. Give empty evidence_ids and covered_by arrays. Never copy an unrealized plan into canon.
4. Evidence may only be selected by ID from evidence_catalog. Never retype, edit, merge, or invent a quote or ID.
5. Audit every requirement exactly once. Do not invent requirement IDs or node aliases.
"""
    validation_error = ""
    for attempt in range(1, _MAX_COVERAGE_AUDIT_ATTEMPTS + 1):
        repair = ""
        if validation_error:
            repair = (
                "\nThe previous audit report failed deterministic validation:\n"
                f"{validation_error}\nReturn a corrected complete audit JSON."
            )
        report = _invoke_json_prompt(audit_prompt + repair)
        try:
            return _validate_plan_coverage_report(
                report,
                requirements,
                chapter_text,
                patch,
                evidence_catalog=evidence_catalog,
            )
        except ValueError as exc:
            validation_error = str(exc)
            if attempt == _MAX_COVERAGE_AUDIT_ATTEMPTS:
                raise ValueError(
                    "Plan coverage audit remained invalid after "
                    f"{attempt} attempts: {validation_error}"
                ) from exc
    raise ValueError("Plan coverage audit produced no result.")


def _validate_monotonic_coverage_repair(previous_patch: dict, repaired_patch: dict) -> None:
    """Coverage repair may append evidence/facts, but may not erase extracted canon."""
    previous_nodes = previous_patch.get("plots_node", [])
    repaired_nodes = repaired_patch.get("plots_node", [])
    if len(repaired_nodes) < len(previous_nodes):
        raise ValueError("Coverage repair may not remove existing plot nodes.")

    stable_fields = ("overview", "importance", "event", "characters_involved", "environment", "foreshadowing")
    for index, previous in enumerate(previous_nodes):
        repaired = repaired_nodes[index]
        for field in stable_fields:
            if repaired.get(field) != previous.get(field):
                raise ValueError(
                    f"Coverage repair may not rewrite plots_node[{index}].{field}; append facts to details or add a node."
                )
        previous_details = previous.get("details", "")
        repaired_details = repaired.get("details", "")
        if previous_details not in repaired_details:
            raise ValueError(
                f"Coverage repair must preserve plots_node[{index}].details verbatim while appending missing facts."
            )

    previous_edges = {
        tuple(edge)
        for edge in previous_patch.get("plots_relationship", [])
        if isinstance(edge, list) and len(edge) == 3
    }
    repaired_edges = {
        tuple(edge)
        for edge in repaired_patch.get("plots_relationship", [])
        if isinstance(edge, list) and len(edge) == 3
    }
    if not previous_edges.issubset(repaired_edges):
        raise ValueError("Coverage repair may not remove existing C_P relationships.")


def _request_coverage_repair(
    base_prompt: str,
    current_patch: dict,
    missing_coverage: list[dict],
    chapter_payload: dict,
) -> dict:
    """Request one monotonic full-patch repair, with two bounded structural attempts."""
    prompt = f"""
{base_prompt}

The post-hoc coverage gate found planned facts that ARE supported by exact chapter prose but are missing from the patch:
{json.dumps(missing_coverage, ensure_ascii=False)}

Current structurally valid patch:
{json.dumps(current_patch, ensure_ascii=False)}

Return a corrected full patch. Preserve every existing node in the same array position, preserve its overview and all non-details fields exactly, keep its old details verbatim, and only append the missing prose-grounded fact to details or append a new scene-anchored node. Preserve every existing edge. Do not add any plan that lacks prose evidence.
"""
    validation_error = ""
    for attempt in range(1, 3):
        repair = ""
        if validation_error:
            repair = (
                "\nThe previous coverage repair was rejected:\n"
                f"{validation_error}\nReturn a corrected monotonic full patch."
            )
        candidate = _invoke_json_prompt(prompt + repair)
        candidate.setdefault("plots_node", [])
        candidate.setdefault("plots_relationship", [])
        candidate.setdefault("summary", "")
        try:
            candidate = _validate_canon_plot_patch(candidate, chapter_payload)
            _validate_monotonic_coverage_repair(current_patch, candidate)
            return candidate
        except ValueError as exc:
            validation_error = str(exc)
            if attempt == 2:
                raise ValueError(
                    f"Plot canon coverage repair remained invalid after {attempt} attempts: {validation_error}"
                ) from exc
    raise ValueError("Plot canon coverage repair produced no result.")


def _attach_plan_coverage_evidence(patch: dict, coverage: list[dict]) -> dict:
    """Persist exact prose spans on covered C_P nodes so retrieval cannot lose the fact."""
    nodes = patch.get("plots_node", [])
    for node in nodes:
        if isinstance(node, dict):
            node["canon_evidence"] = []
    for item in coverage:
        if item.get("status") != "realized_covered":
            continue
        evidence_record = {
            "requirement_id": item["requirement_id"],
            "source": item["source"],
            "prose_evidence": list(item["evidence"]),
        }
        for alias in item["covered_by"]:
            node_index = int(alias.rsplit("_", 1)[1])
            records = nodes[node_index].setdefault("canon_evidence", [])
            if evidence_record not in records:
                records.append(dict(evidence_record))
    patch["plan_coverage_contract"] = 1
    patch["plan_coverage"] = coverage
    return patch


def build_canon_plot_patch(
    chapter_text: str,
    chapter_payload: dict,
    trace_logger=None,
    event_index=None,
) -> dict:
    trace = get_trace_logger(trace_logger)
    plot_refs = [parse_plot_ref(item) for item in chapter_payload.get("refs", {}).get("plots", [])]
    character_refs = [parse_character_ref(item) for item in chapter_payload.get("refs", {}).get("characters", [])]
    environment_refs = [parse_environment_ref(item) for item in chapter_payload.get("refs", {}).get("environments", [])]
    historical_plot_refs = [
        item
        for item in chapter_payload.get("refs", {}).get("plots", [])
        if isinstance(item, str) and parse_plot_ref(item) is not None
    ]
    relationship_example = (
        [[historical_plot_refs[0], "new_node_0", "e_cas"]]
        if historical_plot_refs
        else []
    )
    prompt = f"""
You are the Plot Manager for this novel (managing the Causal Plot Chain C_P).
Task: Extract the canon facts that actually occurred in this chapter's prose and output the plot patch for this chapter.

chapter_payload:
{json.dumps(chapter_payload, ensure_ascii=False)}

normalized_plot_ref_ids: {[item for item in plot_refs if item is not None]}
normalized_character_ids: {[item for item in character_refs if item is not None]}
normalized_environment_ids: {[item for item in environment_refs if item is not None]}

Chapter text:
{chapter_text}

Output JSON:
{{
  "chapter_no": {chapter_payload.get("chapter_no", 1)},
  "summary": "Actual plot summary of this chapter",
  "plots_node": [
    {{
      "chapter": {chapter_payload.get("chapter_no", 1)},
      "overview": "one-sentence plot summary",
      "details": "fairly complete factual description",
      "importance": "main",
      "event": "{chapter_payload.get('chapter_goal', '')}",
      "characters_involved": [],
      "environment": "must be non-empty, a single physical scene name",
      "foreshadowing": false
    }}
  ],
  "plots_relationship": {json.dumps(relationship_example, ensure_ascii=False)}
}}

Rules:
1. Only extract facts actually realized in this chapter's prose. Do not copy planned beats that did not occur.
2. Every plot node is scene-anchored: `environment` must be one non-empty physical scene name, never a list, conceptual rule, region set, or multiple alternatives.
3. Historical relationship endpoints may use only the declared refs in `normalized_plot_ref_ids`; never invent or retrieve another p_K here. New nodes are addressed only by their zero-based aliases new_node_0, new_node_1, ... in array order.
4. Do NOT output e_temp. The application layer deterministically connects adjacent inserted nodes to the chronological backbone.
5. e_cas must point from a genuine cause (a declared historical p_K or an earlier new_node_N) to a current new_node_N consequence. Mere chronology is not causality.
6. e_res has exactly this direction and shape: [new_node_N payoff, historical p_K setup, "e_res"]. Its target must be a declared historical foreshadowing node actually paid off in this prose. Never reverse it and never use e_res merely because a thread is mentioned.
7. Edge types are only e_cas / e_res; old names sequence / caused_by / resolves are forbidden.
8. `importance` must be exactly `main` or `side`; `foreshadowing` must be a JSON boolean.
9. Do not output explanatory text.
"""
    validation_error = ""
    patch = None
    for attempt in range(1, 4):
        repair_instruction = ""
        if validation_error:
            repair_instruction = (
                "\nYour previous patch was rejected by the deterministic validator:\n"
                f"{validation_error}\nReturn a corrected full JSON patch now."
            )
        candidate = _invoke_json_prompt(prompt + repair_instruction)
        candidate.setdefault("plots_node", [])
        candidate.setdefault("plots_relationship", [])
        candidate.setdefault("summary", "")
        try:
            patch = _validate_canon_plot_patch(candidate, chapter_payload)
            break
        except ValueError as exc:
            validation_error = str(exc)
            if attempt == 3:
                raise ValueError(
                    f"Plot canon patch remained invalid after {attempt} attempts: {validation_error}"
                ) from exc
    if patch is None:
        raise ValueError("Plot canon patch validation produced no result.")

    coverage_requirements = _build_plan_coverage_requirements(chapter_payload)
    coverage: list[dict] = []
    coverage_repairs = 0
    if coverage_requirements:
        while True:
            coverage = _audit_plan_coverage(
                chapter_text,
                coverage_requirements,
                patch,
            )
            missing_coverage = [
                item for item in coverage if item.get("status") == "realized_missing"
            ]
            if not missing_coverage:
                break
            if coverage_repairs >= _MAX_COVERAGE_REPAIRS:
                missing_ids = [item["requirement_id"] for item in missing_coverage]
                raise ValueError(
                    "Plot canon patch still omits prose-realized planning requirements "
                    f"after {_MAX_COVERAGE_REPAIRS} repairs: {missing_ids}"
                )
            patch = _request_coverage_repair(
                prompt,
                patch,
                missing_coverage,
                chapter_payload,
            )
            coverage_repairs += 1

    patch = _attach_plan_coverage_evidence(patch, coverage)
    trace.log(
        "plot_canon_patch",
        agent="PlotAgent",
        phase="plot_canon",
        event_index=event_index,
        chapter_no=chapter_payload.get("chapter_no"),
        content=patch.get("summary", ""),
        payload={
            "patch": patch,
            "refs": chapter_payload.get("refs", {}),
            "coverage_requirement_count": len(coverage_requirements),
            "coverage_realized_count": sum(
                item.get("status") == "realized_covered" for item in coverage
            ),
            "coverage_not_realized_count": sum(
                item.get("status") == "not_realized" for item in coverage
            ),
            "coverage_repair_count": coverage_repairs,
        },
        status="ok",
    )
    return patch
