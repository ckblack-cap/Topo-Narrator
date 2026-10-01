
import json
import os
from collections import defaultdict
from typing import Any

from data import data_path


def _load_events(events_path: str) -> list[dict]:
    if not os.path.exists(events_path):
        return []
    out = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _truncate(text: Any, n: int = 200) -> str:
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False) if text is not None else ""
    text = text.strip().replace("\n", " ")
    if len(text) <= n:
        return text
    return text[:n] + " …"


def _format_phase_summary(events_in_event: list[dict]) -> list[dict]:
    phases = ["reason_inquire", "reason_blueprint", "reason_finalize"]
    rows = []
    for phase in phases:
        decision_seq_to_step: dict[int, int] = {}
        step_idx = 0
        for e in events_in_event:
            if (e.get("event_type") == "agent_decision"
                and e.get("phase") == phase
                and e.get("agent") == "EventPlanner"):
                step_idx += 1
                decision_seq_to_step[e.get("seq")] = step_idx
        steps = step_idx

        tool_calls_count = 0
        tools = []
        for e in events_in_event:
            if e.get("event_type") == "agent_tool_call" and e.get("phase") == phase:
                tool_calls_count += 1
                pl = e.get("payload", {}) or {}
                tname = pl.get("tool_name") or e.get("content", "")
                args = pl.get("args", {}) or {}
                req_key = ""
                if isinstance(args, dict):
                    rj = args.get("requirement_json", "")
                    try:
                        if isinstance(rj, str) and rj:
                            parsed = json.loads(rj) or {}
                            req_key = parsed.get("requirement_key", "")
                    except Exception:
                        req_key = ""
                tools.append(f"{tname}({req_key})" if req_key else tname)

        manager_replies = sum(
            1 for e in events_in_event
            if e.get("event_type") == "agent_response" and e.get("phase") == phase
        )

        artifact_count = 0
        nudge_steps: list[int] = []
        force_steps: list[int] = []
        for e in events_in_event:
            if e.get("phase") != phase:
                continue
            et = e.get("event_type")
            if et == "phase_artifact":
                artifact_count += 1
            elif et == "phase_soft_nudge":
                step_no = (e.get("payload", {}) or {}).get("step_count")
                nudge_steps.append(step_no or 0)
            elif et == "phase_force_advance":
                step_no = (e.get("payload", {}) or {}).get("step_count")
                force_steps.append(step_no or 0)
        outcome = []
        if artifact_count:
            outcome.append(f"✅ artifact ×{artifact_count}")
        if nudge_steps:
            outcome.append(f"💡 soft_nudge@step{','.join(map(str, nudge_steps))}")
        if force_steps:
            outcome.append(f"⚠️ force_advance@step{','.join(map(str, force_steps))}")

        red_flag = ""
        if steps >= 25 and tool_calls_count == 0:
            red_flag = " 🚩 0 tool calls (LLM looped without invoking managers)"

        rows.append({
            "phase": phase,
            "steps": steps,
            "tools": tools,
            "tool_calls_count": tool_calls_count,
            "manager_replies": manager_replies,
            "outcome": outcome,
            "red_flag": red_flag,
        })
    return rows


def _extract_validator_history(events_in_event: list[dict]) -> list[dict]:
    return [
        {
            "decision": (e.get("payload", {}) or {}).get("decision"),
            "errors": (e.get("payload", {}) or {}).get("errors", []) or [],
        }
        for e in events_in_event
        if e.get("event_type") == "validator_result"
    ]


def _extract_proposals(events_in_event: list[dict]) -> list[dict]:
    out = []
    for e in events_in_event:
        if e.get("event_type") != "entity_proposal":
            continue
        pl = e.get("payload", {}) or {}
        prop = pl.get("proposal") or {}
        recommended = prop.get("recommended") if isinstance(prop, dict) else {}
        name = ""
        if isinstance(recommended, dict):
            name = recommended.get("name", "") or recommended.get("temp_name", "") or ""
        if not name and isinstance(prop, dict):
            name = prop.get("name", "") or prop.get("title", "") or ""
        out.append({
            "agent": e.get("agent", ""),
            "temp_id": pl.get("temp_id", ""),
            "name": name,
            "type": pl.get("type", ""),
        })
    return out


def _extract_chapter_blocks(events_in_event: list[dict]) -> list[dict]:
    by_ch = defaultdict(list)
    for e in events_in_event:
        cn = e.get("chapter_no")
        if cn:
            by_ch[cn].append(e)
    out = []
    for ch_no in sorted(by_ch.keys()):
        evts = by_ch[ch_no]
        write_start = next((e for e in evts if e.get("event_type") == "chapter_write_start"), None)
        ctx = next(
            (e for e in evts
             if e.get("event_type") == "agent_decision" and e.get("phase") == "chapter_context"),
            None,
        )
        write_done = next((e for e in evts if e.get("event_type") == "chapter_write_done"), None)
        canon = next((e for e in evts if e.get("event_type") == "plot_canon_patch"), None)
        graph_done = next((e for e in evts if e.get("event_type") == "graph_update_done"), None)
        out.append({
            "chapter_no": ch_no,
            "write_start": write_start,
            "context": ctx,
            "write_done": write_done,
            "canon": canon,
            "graph_done": graph_done,
        })
    return out


def _render_event_section(event_idx: int, events_in_event: list[dict], abstract: str = "") -> list[str]:
    lines = [f"## Event {event_idx}", ""]
    if abstract:
        lines.append(f"**Abstract**: {_truncate(abstract, 240)}")
        lines.append("")
    lines.append("### EventPlanner stages")

    phase_rows = _format_phase_summary(events_in_event)
    lines.append("| Phase | LLM Steps | Tool Calls | Mgr Replies | Outcome |")
    lines.append("|---|---|---|---|---|")
    for row in phase_rows:
        tools_str = ", ".join(row["tools"]) if row["tools"] else "(none)"
        outcome_str = " / ".join(row["outcome"]) if row["outcome"] else ""
        tool_count_str = f"**{row['tool_calls_count']}** ({tools_str})" if row['tool_calls_count'] else "**0** (none)"
        red = row.get("red_flag", "")
        lines.append(
            f"| {row['phase']} | {row['steps']} | {tool_count_str} | {row['manager_replies']} | {outcome_str}{red} |"
        )
    lines.append("")

    proposals = _extract_proposals(events_in_event)
    if proposals:
        lines.append("**Proposal outputs**:")
        for p in proposals:
            extra = f" (by {p['agent']})" if p["agent"] else ""
            name_str = f" → name=`{p['name']}`" if p["name"] else ""
            lines.append(f"- `{p['temp_id']}` [{p['type']}]{name_str}{extra}")
        lines.append("")

    validator_history = _extract_validator_history(events_in_event)
    if validator_history:
        lines.append("**Validator history**:")
        for i, v in enumerate(validator_history, 1):
            lines.append(f"- Round {i}: **{v['decision']}** ({len(v['errors'])} errors)")
            for err in v["errors"][:5]:
                lines.append(f"    - {_truncate(err, 200)}")
            if len(v["errors"]) > 5:
                lines.append(f"    - ... total {len(v['errors'])} entries")
        lines.append("")

    backfills = [e for e in events_in_event if e.get("event_type") == "seed_subgraph_backfill"]
    for bf in backfills:
        items = (bf.get("payload", {}) or {}).get("backfilled", []) or []
        lines.append(f"**Backfill** (reverse-fill seed_subgraph): {', '.join(items) or '(none)'}")
    if backfills:
        lines.append("")

    return lines


def _render_chapter_block(block: dict) -> list[str]:
    cn = block["chapter_no"]
    lines = [f"### Chapter {cn}", ""]
    ctx = block.get("context")
    write_start = block.get("write_start")
    write_done = block.get("write_done")
    canon = block.get("canon")
    graph_done = block.get("graph_done")

    lines.append("**Writer input**:")
    if write_start:
        lines.append(f"- `pure_plot`: {_truncate(write_start.get('content', ''), 200)}")
        wpl = write_start.get("payload", {}) or {}
        refs = wpl.get("refs", {}) or {}
        if refs:
            ref_summary = " / ".join(
                f"{k}={refs.get(k, []) or []}" for k in ("characters", "environments", "plots")
            )
            lines.append(f"- `refs`: {ref_summary}")
    if ctx:
        pl = ctx.get("payload", {}) or {}
        ci = pl.get("characters_info", "")
        ei = pl.get("environments_info", "")
        of = pl.get("open_foreshadowing", []) or []
        rh = pl.get("relationship_history", []) or []
        lines.append(f"- `characters_info` (first 240 characters): {_truncate(ci, 240)}")
        lines.append(f"- `environments_info` (first 240 characters): {_truncate(ei, 240)}")
        lines.append(f"- `open_foreshadowing`: {len(of)} entries")
        lines.append(f"- `relationship_history`: {len(rh)} entries")
    lines.append("")

    if write_done:
        body = write_done.get("content", "") or ""
        lines.append("**Writer output**:")
        lines.append(f"- Body length: {len(body)} characters")
        lines.append(f"- Preview: {_truncate(body, 240)}")
        lines.append("")

    if canon:
        cpl = canon.get("payload", {}) or {}
        patch = cpl.get("patch", {}) or cpl
        plots_node = patch.get("plots_node", []) or []
        chars_node = patch.get("characters_node", []) or []
        envs_node = patch.get("environments_node", []) or []
        lines.append(
            f"**Canon Patch**: {len(chars_node)} characters / {len(envs_node)} environments / {len(plots_node)} foreshadowing nodes"
        )
        if patch.get("summary"):
            lines.append(f"- summary: {_truncate(patch.get('summary'), 200)}")
        lines.append("")

    if graph_done:
        gpl = graph_done.get("payload", {}) or {}
        cn_added = gpl.get("characters_name", []) or []
        en_added = gpl.get("envs_name", []) or []
        if isinstance(cn_added, list) and cn_added:
            lines.append(f"**G_C characters**: {', '.join(map(str, cn_added))}")
        if isinstance(en_added, list) and en_added:
            lines.append(f"**T_E environments**: {', '.join(map(str, en_added))}")
        if cn_added or en_added:
            lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def generate_trace_digest(premise_id: str, run_id: str) -> str | None:
    try:
        run_dir = os.path.join(data_path(), str(premise_id), "agent_runs", str(run_id))
        events_path = os.path.join(run_dir, "events.jsonl")
        events = _load_events(events_path)
        if not events:
            print(f">>> trace_digest: events.jsonl is empty or missing; skipping ({events_path})")
            return None

        run_start = next((e for e in events if e.get("event_type") == "run_start"), None)
        run_done = next((e for e in events if e.get("event_type") == "run_done"), None)

        by_event = defaultdict(list)
        for e in events:
            ei = e.get("event_index")
            if ei is not None:
                by_event[ei].append(e)

        storyline_abstracts = {}
        for e in events:
            if e.get("event_type") == "event_planning_start":
                storyline_abstracts[e.get("event_index")] = e.get("content", "")

        lines = ["# Topo-Narrator Run Digest", ""]
        if run_start:
            pl = run_start.get("payload", {}) or {}
            ablation = pl.get("ablation", {}) or {}
            lines.append(f"- **Premise ID**: `{run_start.get('premise_id', premise_id)}`")
            lines.append(f"- **Run ID**: `{run_id}`")
            lines.append(f"- **Started**: {run_start.get('time', '')}")
            lines.append(f"- **Max Events**: {pl.get('max_events', '?')}")
            lines.append(
                f"- **Ablation**: foreshadowing={ablation.get('enable_foreshadowing_injection')}, "
                f"hij={ablation.get('enable_hij_injection')}"
            )
            lines.append("")
        lines.append("---")
        lines.append("")

        for ei in sorted(by_event.keys()):
            evts_in = by_event[ei]
            lines.extend(_render_event_section(ei, evts_in, storyline_abstracts.get(ei, "")))
            for block in _extract_chapter_blocks(evts_in):
                lines.extend(_render_chapter_block(block))
            lines.append("")

        # ---- Run Summary ----
        total_llm = sum(1 for e in events if e.get("event_type") == "agent_decision")
        total_chapters = sum(1 for e in events if e.get("event_type") == "chapter_write_done")
        total_revises = sum(
            1 for e in events
            if e.get("event_type") == "validator_result"
            and (e.get("payload", {}) or {}).get("decision") == "REVISE"
        )
        total_backfills = sum(1 for e in events if e.get("event_type") == "seed_subgraph_backfill")
        errors_count = sum(1 for e in events if e.get("event_type") == "error")
        force_adv_count = sum(1 for e in events if e.get("event_type") == "phase_force_advance")
        nudge_count = sum(1 for e in events if e.get("event_type") == "phase_soft_nudge")
        total_tool_calls = sum(1 for e in events if e.get("event_type") == "agent_tool_call")
        total_mgr_replies = sum(1 for e in events if e.get("event_type") == "agent_response")

        lines.append("## Run Summary")
        lines.append(f"- LLM calls (agent_decision events): **{total_llm}**")
        lines.append(f"- Tool calls (Plan→Manager): **{total_tool_calls}**")
        lines.append(f"- Manager replies returned: **{total_mgr_replies}**")
        if total_llm and total_tool_calls / max(total_llm, 1) < 0.10:
            lines.append(f"  - 🚩 Tool-call ratio < 10% — LLM may be looping without invoking managers")
        lines.append(f"- Chapters generated: **{total_chapters}**")
        lines.append(f"- Validator REVISEs: **{total_revises}**")
        lines.append(f"- seed_subgraph backfills: **{total_backfills}**")
        lines.append(f"- Phase soft_nudges: **{nudge_count}**")
        lines.append(f"- Phase force_advances: **{force_adv_count}**")
        lines.append(f"- Errors: **{errors_count}**")
        if run_done:
            pl = run_done.get("payload", {}) or {}
            lines.append(f"- Final global_chapter_num: {pl.get('global_chapter_num', '?')}")
            lines.append(f"- Run finished at: {run_done.get('time', '')}")

        digest_path = os.path.join(run_dir, "digest.md")
        with open(digest_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f">>> trace_digest: digest generated → {digest_path}")
        return digest_path
    except Exception as exc:
        print(f">>> trace_digest generation failed (main workflow continues): {exc}")
        return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m tools.trace_digest <premise_id> <run_id>")
        sys.exit(1)
    generate_trace_digest(sys.argv[1], sys.argv[2])
