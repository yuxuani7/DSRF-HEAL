from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


Message = Dict[str, Any]


@dataclass
class LLMConfig:
    name: str
    provider: str
    base_url: str
    api_key: str
    model: str
    default_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HEALSample:
    dataset_name: str
    task_id: str
    prompt: str
    messages: List[Message]
    meta: Dict[str, Any]


@dataclass
class HEALDataset:
    name: str
    source_path: Path
    samples: List[HEALSample] = field(default_factory=list)


@dataclass
class LoadError:
    dataset_name: str
    source_path: str
    error_type: str
    message: str
    traceback: Optional[str] = None


@dataclass
class LLMCallError:
    type: str
    message: str
    traceback: Optional[str] = None


@dataclass
class LLMCallResult:
    text: str
    finish_reason: Optional[str]
    latency_ms: int
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    raw_chunk_count: int
    error: Optional[LLMCallError] = None

