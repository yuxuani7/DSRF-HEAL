import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from src.types import LLMConfig


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _resolve_env_placeholders(text: str) -> Tuple[str, List[str]]:
    missing = []

    def _replace(match: re.Match) -> str:
        env_name = match.group(1)
        env_value = os.environ.get(env_name)
        if env_value is None:
            missing.append(env_name)
            return ""
        return env_value

    return ENV_PATTERN.sub(_replace, text), missing


def load_llm_configs(config_path: Path) -> Dict[str, LLMConfig]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pyyaml") from exc

    if not config_path.exists():
        raise FileNotFoundError("Config file not found: {0}".format(config_path))

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    llms = raw.get("llms")
    if not isinstance(llms, list):
        raise ValueError("Invalid llms config: `llms` must be a list")

    parsed: Dict[str, LLMConfig] = {}
    for idx, item in enumerate(llms):
        if not isinstance(item, dict):
            raise ValueError("Invalid llm entry at index {0}".format(idx))
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("Missing `name` in llm entry index {0}".format(idx))
        if name in parsed:
            raise ValueError("Duplicate llm name: {0}".format(name))

        provider = str(item.get("provider", "")).strip()
        if provider != "openai_compatible":
            raise ValueError(
                "Unsupported provider `{0}` for llm `{1}`".format(provider, name)
            )

        base_url = str(item.get("base_url", "")).strip()
        api_key_raw = str(item.get("api_key", "")).strip()
        model = str(item.get("model", "")).strip()
        if not base_url or not api_key_raw or not model:
            raise ValueError(
                "LLM `{0}` must include base_url, api_key and model".format(name)
            )

        api_key, missing_envs = _resolve_env_placeholders(api_key_raw)
        if missing_envs:
            raise ValueError(
                "LLM `{0}` missing env vars for api_key: {1}".format(
                    name, ", ".join(sorted(set(missing_envs)))
                )
            )
        if not api_key:
            raise ValueError("LLM `{0}` resolved empty api_key".format(name))

        default_params = item.get("default_params") or {}
        if not isinstance(default_params, dict):
            raise ValueError(
                "LLM `{0}` has invalid default_params, expected object".format(name)
            )

        parsed[name] = LLMConfig(
            name=name,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            default_params=default_params,
        )
    return parsed

