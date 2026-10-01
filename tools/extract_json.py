import re
import json
from typing import Optional, Dict
import json5
def extract_json(llm_output: str, enable_llm_correction: bool = True ) -> Optional[Dict]:
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, llm_output, re.DOTALL)

    if matches:
        json_str = matches[0].strip()
    else:
        json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
        matches = re.findall(json_pattern, llm_output)
        if not matches:
            print("No JSON content found")
            return None
        json_str = matches[0].strip()

    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        try:
            return json5.loads(json_str)
        except Exception as e2:
            print(f">>> JSON5 parsing failed: {e2}")

            if enable_llm_correction:
                print(">>> Starting LLM-assisted JSON repair")
                try:
                    task_prompt = f"""
                    You are a strict JSON syntax-correction expert. You must complete this task following the rules below exactly:
                    ### Task requirements
                    1. You receive a JSON string with syntax errors. Fix every issue that violates the JSON spec, so that the corrected JSON can be parsed successfully by a standard JSON parser.
                    2. Fixes include but are not limited to:
                       - Removing trailing commas in objects/arrays;
                       - Replacing all single quotes with double quotes (JSON allows only double quotes);
                       - Ensuring every property name is wrapped in double quotes;
                       - Fixing missing closing braces / brackets;
                       - Correcting bad escape characters, newlines, and other special-character mistakes;
                       - Keeping the original data (text, numbers, structure) 100% unchanged — only syntax is corrected.

                    ### Output rules (must be followed strictly, otherwise the task fails)
                    1. Output only the corrected pure JSON text — no extra content (no explanation, no markdown code block, no comments, no whitespace other than necessary newlines).
                    2. Do not output any ```json ``` wrapping.
                    3. Do not add any irrelevant text (e.g., "Here is the corrected JSON:").
                    4. Ensure the JSON is compact and well-formed, with correct array / object nesting.

                    ### The malformed JSON string to repair (json5 parse error: {e2})
                    {json_str}""".strip()

                    from agents import AgentGlobalConfig, DEFAULT_MODEL_NAME
                    client = AgentGlobalConfig.GPTCLIENT
                    response = client.chat.completions.create(
                        model=DEFAULT_MODEL_NAME,
                        messages=[
                            {"role": "system",
                             "content": "You are a strict JSON syntax-correction expert. Output only standards-compliant JSON text — nothing else."},
                            {"role": "user", "content": task_prompt}
                        ],
                        temperature=0.0,
                        timeout=120,
                    )

                    corrected_json_str = response.choices[0].message.content.strip()
                    if corrected_json_str.startswith("```json"):
                        corrected_json_str = corrected_json_str[7:-3].strip()

                    subgraph = json.loads(corrected_json_str)
                    print(">>> LLM-assisted JSON repair succeeded; parsing complete")
                    return subgraph
                except Exception as llm_e:
                    print(f">>> LLM-assisted repair also failed: {llm_e}")
                    return None

