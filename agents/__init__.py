import os
import openai

# =========================================================================
# =========================================================================
DEFAULT_MODEL_NAME = os.getenv("TOPO_DEFAULT_MODEL", "gpt-4o-mini")

DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0


def _validated_float_env(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return float(default)
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be numeric; got {raw_value!r}.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"Environment variable {name} must be within [{minimum}, {maximum}]; got {value}."
        )
    return value


def _validated_int_env(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return int(default)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer; got {raw_value!r}.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"Environment variable {name} must be within [{minimum}, {maximum}]; got {value}."
        )
    return value


def _raw_sdk_runtime_options() -> dict:
    return {
        "timeout": _validated_float_env(
            "TOPO_LLM_REQUEST_TIMEOUT_SECONDS",
            DEFAULT_REQUEST_TIMEOUT_SECONDS,
            minimum=1.0,
            maximum=600.0,
        ),
        "max_retries": _validated_int_env(
            "TOPO_LLM_SDK_MAX_RETRIES",
            0,
            minimum=0,
            maximum=3,
        ),
    }


class _LazyOpenAIClient:

    def __init__(self, api_key_env: str, base_url_env: str, default_base_url: str):
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.default_base_url = default_base_url
        self._client = None

    def _resolve(self):
        if self._client is None:
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Environment variable {self.api_key_env} is missing; the LLM provider cannot be called."
                )
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=os.getenv(self.base_url_env, self.default_base_url),
                **_raw_sdk_runtime_options(),
            )
        return self._client

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


def _require_api_key(env_name: str) -> str:
    api_key = os.getenv(env_name)
    if not api_key:
        raise RuntimeError(f"Environment variable {env_name} is missing; cannot create the LLM client.")
    return api_key


class AgentGlobalConfig:
    GPTCLIENT = _LazyOpenAIClient(
        "GPT_API_KEY",
        "GPT_BASE_URL",
        "https://chatgpt.com",
    )

