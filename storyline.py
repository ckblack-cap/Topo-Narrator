import os
import json
import re
import time
from agents import AgentGlobalConfig, DEFAULT_MODEL_NAME
from data import set_file
from tools.agent_event_logger import AgentEventLogger, get_trace_logger, make_trace_run_id

client = AgentGlobalConfig.GPTCLIENT



DATA_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

OUTLINE_STAGE_ORDER = ("Setup", "Confrontation", "Resolution")
OUTLINE_EVENT_MIN_CHARS = 10
OUTLINE_EVENT_MAX_CHARS = 420
OUTLINE_EVENT_MAX_SENTENCES = 2


PROMPT_GENERATE_OUTLINE_LONG = """
Role: You are a master story architect and senior novelist.

Task: Based on the given PREMISE, create a plot outline for a novel (paper experiment configuration: N=3 Events × 2 chapters per Event ≈ 13000 words).
Structure: Follow the classic three-act structure (Setup / Confrontation / Resolution).
**Plan exactly one key event per act, for a total of 3 events; the levels must be clear, the logic coherent, and each event must leave hooks for the next.**

Output format:
Return only a **strict JSON array** containing **exactly 3 objects** (one per act). No Markdown, no extra text.
Format:
[
    {{
        "stage": "Setup",
        "event": "Summarize the core event of the setup act in one clear, concise sentence."
    }},
    {{
        "stage": "Confrontation",
        "event": "Summarize the core event of the confrontation act in one clear, concise sentence."
    }},
    {{
        "stage": "Resolution",
        "event": "Summarize the core event of the resolution act in one clear, concise sentence."
    }}
]

Constraints:
1. Use abstract narrative descriptions.
2. Keep each `event` at a high level: 1-2 concise sentences and no more than 420 characters. Do not expand it into chapter beats or prose.
3. Participants may be named because an event abstract may identify its participants. If you introduce a name, keep that exact name and narrative role consistent across all three acts; never rename or replace the same participant between acts.
4. Ensure the events strictly follow the premise, and that the three events have a clear causal succession.

PREMISE:
{premise}
"""



PROMPT_EVALUATE_LONG = """
Role: You are a cold, uncompromising senior novel editor.

Task: Evaluate the following three-act story outline based on the HANNA 2022 automatic story generation evaluation metrics.
The target length of this configuration is about 13000 words (3 Events × 2 chapters). It is not required to support a long-form serial, but the three acts must be clearly connected, with strong conflict and payoff. If the outline falls short, simply provide concrete improvement suggestions.

Evaluation criteria:
1. Relevance: Does it strictly follow the premise?
2. Coherence: Is the causal logic between the three acts free of holes?
3. Empathy: Do the abstract descriptions imply strong character motivation?
4. Surprise: Does it avoid clichés and offer unexpected turns?
5. Creativity: Is the narrative approach novel and engaging?
6. Complexity: Does it show depth beyond a simple linear path?

Output format:
Return only a single JSON object.
{{
    "score": <0-10>,
    "critique": "Based on the criteria above, give a sharp, specific paragraph pointing out the weaknesses.",
    "suggestion": "Give actionable, explicit revision instructions for the specific weaknesses."
}}

PREMISE: {premise}
CURRENT OUTLINE: {outline}
"""



PROMPT_REVISE = """
Role: You are a top-tier Script Doctor.

Task: Rewrite the story outline based on the editor's critique.
Keep the JSON format strictly identical to the original version, but improve each "event" summary so it is more coherent, more creative, and more emotionally resonant.
Each event must remain a high-level summary of 1-2 concise sentences (no more than 420 characters). Preserve participant names and narrative roles consistently across Setup, Confrontation, and Resolution.

Output format: Return only the revised JSON array.

PREMISE: {premise}
ORIGINAL OUTLINE: {outline}
EDITOR'S CRITIQUE: {critique}
EDITOR'S SUGGESTIONS: {suggestion}
"""


PROMPT_CONCEPT_TAXONOMY = """
Role: You are the worldbuilding architect for this novel.

Task: Based on the PREMISE and the three-act outline, distill the 1-2 most critical [worldbuilding / rule taxonomies]
to serve as seeds for the World Concept Tree (T_conc).
Requirements:
1. Concepts must be strongly tied to the premise (e.g., disaster tiers, factional organizations, energy systems, safe-zone classifications).
2. Each subtree should be no more than 2 levels deep.
3. The attributes field on child nodes should provide inheritable key/value pairs where possible (e.g., tier, energy_source, access_rule).
4. Node names must be short, unique, and usable as retrieval keys downstream.

Output format: Return only a JSON array (**strict JSON**, no Markdown or explanatory text).
Each element has the structure:
{{
  "name": "top-level concept name",
  "description": "short description",
  "attributes": {{"key": "value"}},
  "children": [
    {{"name": "...", "description": "...", "attributes": {{}}, "children": []}}
  ]
}}

PREMISE:
{premise}

OUTLINE:
{outline}
"""


def generate_concept_taxonomy(
    premise_text: str,
    outline,
    trace_logger=None,
) -> list[dict]:
    prompt = PROMPT_CONCEPT_TAXONOMY.format(premise=premise_text, outline=json.dumps(outline, ensure_ascii=False))
    data = _run_json_stage(
        "concept_taxonomy",
        prompt,
        expected_type=list,
        trace_logger=trace_logger,
    )
    return data if isinstance(data, list) else []


def seed_world_concept_tree(
    premise_id: str,
    premise_text: str,
    outline,
    trace_logger=None,
) -> None:
    trace = get_trace_logger(trace_logger)
    taxonomy = generate_concept_taxonomy(premise_text, outline, trace_logger=trace)
    if not taxonomy:
        raise RuntimeError("T_conc taxonomy is empty; cannot continue without world-rule memory.")
    trace.log(
        "global_planner_concept_taxonomy_seed_start",
        agent="GlobalPlanner",
        phase="global_planning",
        content="Starting to write the concept taxonomy to T_conc",
        payload={"root_count": len(taxonomy)},
        status="running",
    )
    try:
        from tools.environment_graph_manager import EnvironmentTreeManager
        manager = EnvironmentTreeManager(premise_id)
        result = manager.init_world_concept_tree(taxonomy)
        if not isinstance(result, str) or result.strip().lower().startswith(("error", "\u9519\u8bef")):
            raise RuntimeError(f"T_conc initialization failed: {result!r}")
        if manager.save_env_graph() is not True:
            raise OSError(f"Failed to save T_conc: premise={premise_id}")
        print(f"🌳 T_conc seeded: {result}")
        trace.log(
            "global_planner_concept_taxonomy_seed_done",
            agent="GlobalPlanner",
            phase="global_planning",
            content="Concept taxonomy written to T_conc",
            payload={"root_count": len(taxonomy), "manager_result": result},
            status="ok",
        )
    except Exception as e:
        print(f"⚠️ T_conc seeding error: {e}")
        trace.log(
            "global_planner_concept_taxonomy_seed_error",
            agent="GlobalPlanner",
            phase="global_planning",
            content="Failed to write the concept taxonomy to T_conc",
            error=e,
            status="error",
        )
        raise




_TRANSIENT_API_SIGNALS = (
    "429", "502", "503", "504", "bad gateway", "gateway", "timeout",
    "temporarily unavailable", "internalservererror", "apiconnectionerror",
    "ratelimiterror", "rate limit", "model_price_error", "load is saturated", "please try again later",
    "\u8d1f\u8f7d\u5df2\u9971\u548c", "\u8bf7\u7a0d\u540e\u518d\u8bd5",
)


class GlobalPlannerRequestError(RuntimeError):
    """A provider request failed before a valid stage payload was available."""


def _retry_delay_for_attempt(base_delay: float, max_delay: float, attempt: int) -> float:
    """Return bounded exponential backoff after a failed 1-based attempt."""
    if base_delay <= 0 or max_delay <= 0:
        return 0.0
    return min(max_delay, base_delay * (2 ** max(0, attempt - 1)))


def _validated_retry_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer; got {raw!r}.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"Environment variable {name} must be within [{minimum}, {maximum}]; got {value}.")
    return value


def _validated_retry_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be numeric; got {raw!r}.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"Environment variable {name} must be within [{minimum}, {maximum}]; got {value}.")
    return value


def get_completion(
    prompt,
    model=None,
    *,
    trace_logger=None,
    stage: str = "unknown",
    stage_payload: dict | None = None,
):
    if model is None:
        model = DEFAULT_MODEL_NAME
    trace = get_trace_logger(trace_logger)
    attempts = _validated_retry_int("TOPO_GLOBAL_PLANNER_ATTEMPTS", 5, 1, 6)
    retry_delay_base = _validated_retry_float(
        "TOPO_GLOBAL_PLANNER_RETRY_DELAY_SECONDS", 5.0, 0.0, 60.0
    )
    retry_delay_cap = _validated_retry_float(
        "TOPO_GLOBAL_PLANNER_RETRY_MAX_DELAY_SECONDS", 60.0, 0.0, 300.0
    )
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("The LLM returned an empty completion.")
            return content.strip()
        except Exception as e:
            error_blob = (e.__class__.__name__ + " " + str(e)).lower()
            is_transient = any(signal in error_blob for signal in _TRANSIENT_API_SIGNALS)
            if not is_transient or attempt == attempts:
                print(
                    f"❌ API Call Error (stage={stage}, attempt {attempt}/{attempts}, "
                    f"transient={is_transient}): {e}"
                )
                raise GlobalPlannerRequestError(
                    f"Global Planner stage {stage} request failed "
                    f"after {attempt}/{attempts} attempt(s); transient={is_transient}: {e}"
                ) from e
            retry_delay = _retry_delay_for_attempt(
                retry_delay_base,
                retry_delay_cap,
                attempt,
            )
            trace.log(
                "global_planner_request_retry",
                agent="GlobalPlanner",
                phase="global_planning",
                content=f"{stage} request encountered a transient error; preparing a bounded retry",
                payload={
                    **(stage_payload or {}),
                    "stage": stage,
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "retry_delay_seconds": retry_delay,
                },
                error=e,
                status="retry",
            )
            print(
                f"⚠️  Transient API error (stage={stage}, attempt {attempt}/{attempts}); "
                f"retrying in {retry_delay:g} seconds: {e}"
            )
            if retry_delay:
                time.sleep(retry_delay)
    raise RuntimeError(f"Global Planner stage {stage} exhausted without a result.")

def parse_json_response(response_text):
    if not response_text:
        print("❌ JSON Parsing Error. Empty response from LLM (likely API failure).")
        return None
    try:
        print(response_text)
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except json.JSONDecodeError:
        print(f"❌ JSON Parsing Error. Raw content:\n{response_text}")
        return None


def _validate_stage_payload(stage: str, parsed) -> None:
    if stage in {"outline", "revise"}:
        if len(parsed) != 3:
            raise ValueError(f"{stage} must contain exactly three Events; got {len(parsed)}.")
        for index, (item, expected_stage) in enumerate(
            zip(parsed, OUTLINE_STAGE_ORDER),
            start=1,
        ):
            if not isinstance(item, dict):
                raise ValueError(f"{stage}[{index}] must be an object.")
            actual_stage = item.get("stage")
            if actual_stage != expected_stage:
                raise ValueError(
                    f"{stage}[{index}].stage must be exactly {expected_stage!r}; "
                    f"got {actual_stage!r}."
                )
            if not isinstance(item.get("event"), str) or not item["event"].strip():
                raise ValueError(f"{stage}[{index}].event must be a nonempty string.")
            event_text = item["event"].strip()
            if not OUTLINE_EVENT_MIN_CHARS <= len(event_text) <= OUTLINE_EVENT_MAX_CHARS:
                raise ValueError(
                    f"{stage}[{index}].event length must be within "
                    f"[{OUTLINE_EVENT_MIN_CHARS}, {OUTLINE_EVENT_MAX_CHARS}], "
                    f"got {len(event_text)}."
                )
            sentence_boundaries = re.findall(r"[.!?\u3002\uFF01\uFF1F]+", event_text)
            sentence_count = max(1, len(sentence_boundaries))
            if sentence_count > OUTLINE_EVENT_MAX_SENTENCES:
                raise ValueError(
                    f"{stage}[{index}].event must remain a high-level summary, with at most "
                    f"{OUTLINE_EVENT_MAX_SENTENCES} sentences; got about {sentence_count}."
                )
    elif stage == "evaluate":
        score = parsed.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("evaluate.score must be numeric.")
        if not 0 <= float(score) <= 10:
            raise ValueError("evaluate.score must be within [0, 10].")
    elif stage == "concept_taxonomy":
        if not parsed:
            raise ValueError("concept_taxonomy must contain at least one root node.")
        for index, item in enumerate(parsed, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"concept_taxonomy[{index}] must be an object.")
            if not isinstance(item.get("name"), str) or not item["name"].strip():
                raise ValueError(f"concept_taxonomy[{index}].name must be a nonempty string.")


def _run_json_stage(
    stage: str,
    prompt: str,
    *,
    expected_type: type,
    trace_logger=None,
    stage_payload: dict | None = None,
):
    trace = get_trace_logger(trace_logger)
    event_prefix = f"global_planner_{stage}"
    payload = dict(stage_payload or {})
    started_at = time.monotonic()
    trace.log(
        f"{event_prefix}_start",
        agent="GlobalPlanner",
        phase="global_planning",
        content=f"Starting Global Planner stage: {stage}",
        payload=payload,
        status="running",
    )
    try:
        raw = get_completion(
            prompt,
            trace_logger=trace,
            stage=stage,
            stage_payload=payload,
        )
        parsed = parse_json_response(raw)
        if not isinstance(parsed, expected_type):
            raise ValueError(
                f"{stage} must return {expected_type.__name__}; "
                f"got {type(parsed).__name__}."
            )
        _validate_stage_payload(stage, parsed)
    except Exception as exc:
        trace.log(
            f"{event_prefix}_error",
            agent="GlobalPlanner",
            phase="global_planning",
            content=f"Global Planner stage failed: {stage}",
            payload={**payload, "elapsed_seconds": round(time.monotonic() - started_at, 3)},
            error=exc,
            status="error",
        )

        if isinstance(exc, GlobalPlannerRequestError):
            raise
        return None

    done_payload = {
        **payload,
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
    }
    if isinstance(parsed, list):
        done_payload["item_count"] = len(parsed)
    elif isinstance(parsed, dict):
        done_payload["keys"] = list(parsed.keys())
        if "score" in parsed:
            done_payload["score"] = parsed.get("score")
    trace.log(
        f"{event_prefix}_done",
        agent="GlobalPlanner",
        phase="global_planning",
        content=f"Global Planner stage completed: {stage}",
        payload=done_payload,
        status="ok",
    )
    return parsed


def save_to_file(premise_id, outline_data):
    dir_path = os.path.join(DATA_ROOT_DIR, str(premise_id))

    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, "storyline.json")

    if not set_file(file_path, outline_data):
        raise OSError(f"File Save Error: {file_path}")
    print(f"💾 Saved successfully to: {file_path}")



def process_single_premise(premise_id, premise_text, trace_logger=None):
    trace = get_trace_logger(trace_logger)
    print(f"📝 Premise: {premise_text}\n")

    MAX_RETRIES = 3
    TARGET_SCORE = 8.3

    print("... 1. Generating the initial outline ...")
    current_outline = _run_json_stage(
        "outline",
        PROMPT_GENERATE_OUTLINE_LONG.format(premise=premise_text),
        expected_type=list,
        trace_logger=trace,
    )

    if not current_outline:
        raise RuntimeError("Initial outline generation or parsing failed; stopped without reusing an old storyline.json.")


    final_outline = current_outline

    for i in range(MAX_RETRIES):
        print(f"\nEvaluation round: --- Round {i + 1} / {MAX_RETRIES} ---")

        print("... Evaluating ...")
        eval_json = _run_json_stage(
            "evaluate",
            PROMPT_EVALUATE_LONG.format(
                premise=premise_text,
                outline=json.dumps(current_outline, ensure_ascii=False),
            ),
            expected_type=dict,
            trace_logger=trace,
            stage_payload={"round": i + 1, "max_rounds": MAX_RETRIES},
        )

        if not eval_json:
            print("⚠️ Evaluation returned an invalid format; stopping refinement and saving the current version.")
            break

        score = eval_json.get('score', 0)
        critique = eval_json.get('critique', 'No critique provided.')
        suggestion = eval_json.get('suggestion', '')

        print(f"🧐 Evaluation score:{score}/10")

        if score >= TARGET_SCORE:
            print("✨ Quality threshold met!")
            final_outline = current_outline
            break

        if i == MAX_RETRIES - 1:
            print("🛑 Maximum retry count reached; stopping refinement even though the score is below the threshold.")
            final_outline = current_outline
            break

        print("... Score below threshold; rewriting based on feedback ...")
        print(f"👉 Evaluation: {critique}...")

        revised_json = _run_json_stage(
            "revise",
            PROMPT_REVISE.format(
                premise=premise_text,
                outline=json.dumps(current_outline, ensure_ascii=False),
                critique=critique,
                suggestion=suggestion,
            ),
            expected_type=list,
            trace_logger=trace,
            stage_payload={"round": i + 1, "max_rounds": MAX_RETRIES},
        )

        if revised_json:
            print("✅ Revision complete; preparing the next evaluation round.")
            current_outline = revised_json
            final_outline = current_outline
        else:
            print("⚠️ Revised version could not be parsed; keeping the previous version and stopping.")
            break

    print(f"\n💾 Saving final output... (ID: {premise_id})")
    _validate_stage_payload("outline", final_outline)
    save_to_file(premise_id, final_outline)

    seed_world_concept_tree(
        premise_id,
        premise_text,
        final_outline,
        trace_logger=trace,
    )
    return final_outline


def generate_storyline(pid, ptext, trace_logger=None, trace_run_id: str | None = None):
    if trace_logger is None:
        trace_logger = AgentEventLogger(
            pid,
            trace_run_id or make_trace_run_id(str(pid)),
        )
    trace = get_trace_logger(trace_logger)
    os.makedirs(DATA_ROOT_DIR, exist_ok=True)
    trace.log(
        "global_planner_start",
        agent="GlobalPlanner",
        phase="global_planning",
        content=f"Starting the three-act outline for premise={pid}",
        payload={"premise_id": str(pid)},
        status="running",
    )
    try:
        outline = process_single_premise(pid, ptext, trace_logger=trace)
        if not outline:
            raise RuntimeError("Global Planner did not return a valid outline.")
        trace.log(
            "global_planner_done",
            agent="GlobalPlanner",
            phase="global_planning",
            content=f"Global Planner completed for premise={pid}",
            payload={"premise_id": str(pid), "outline_count": len(outline)},
            status="ok",
        )
        return outline
    except Exception as exc:
        trace.log(
            "global_planner_error",
            agent="GlobalPlanner",
            phase="global_planning",
            content=f"Global Planner failed for premise={pid}",
            payload={"premise_id": str(pid)},
            error=exc,
            status="error",
        )
        raise







