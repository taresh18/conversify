from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import time
import logging

from livekit.agents.llm import function_context, llm
from livekit.agents.llm.function_context import _is_optional_type
import inspect


WhisperModels = Literal[
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
]


TTSModels = Literal[
    "tts-1", 
]

TTSVoices = Literal[
    "af_heart",
    "af_bella"
]


class find_time:
    """A context manager for timing code execution."""
    def __init__(self, label: str):
        self.label = label
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.perf_counter()
        elapsed_ms = (end_time - self.start_time) * 1000
        logging.debug(f"{self.label} took {elapsed_ms:.4f} ms")

ChatModels = Literal[
    'gpt'
]

def build_oai_message(msg: llm.ChatMessage, cache_key: Any):
    oai_msg: dict[str, Any] = {"role": msg.role}

    if msg.name:
        oai_msg["name"] = msg.name

    # add content if provided
    if isinstance(msg.content, str):
        oai_msg["content"] = msg.content
    elif isinstance(msg.content, dict):
        oai_msg["content"] = json.dumps(msg.content)
    elif isinstance(msg.content, list):
        oai_content: list[dict[str, Any]] = []
        for cnt in msg.content:
            if isinstance(cnt, str):
                oai_content.append({"type": "text", "text": cnt})
            elif isinstance(cnt, llm.ChatImage):
                oai_content.append(_build_oai_image_content(cnt, cache_key))

        oai_msg["content"] = oai_content

    # make sure to provide when function has been called inside the context
    # (+ raw_arguments)
    # if msg.tool_calls is not None:
    #     tool_calls: list[dict[str, Any]] = []
    #     oai_msg["tool_calls"] = tool_calls
    #     for fnc in msg.tool_calls:
    #         tool_calls.append(
    #             {
    #                 "id": fnc.tool_call_id,
    #                 "type": "function",
    #                 "function": {
    #                     "name": fnc.function_info.name,
    #                     "arguments": fnc.raw_arguments,
    #                 },
    #             }
    #         )

    # tool_call_id is set when the message is a response/result to a function call
    # (content is a string in this case)
    # if msg.tool_call_id:
    #     oai_msg["tool_call_id"] = msg.tool_call_id

    return oai_msg


def build_oai_function_description(
    fnc_info: function_context.FunctionInfo,
    capabilities: llm.LLMCapabilities | None = None,
) -> dict[str, Any]:
    def build_oai_property(arg_info: function_context.FunctionArgInfo):
        def type2str(t: type) -> str:
            if t is str:
                return "string"
            elif t in (int, float):
                return "number"
            elif t is bool:
                return "boolean"

            raise ValueError(f"unsupported type {t} for ai_property")

        p: dict[str, Any] = {}

        if arg_info.description:
            p["description"] = arg_info.description

        is_optional, inner_th = _is_optional_type(arg_info.type)

        if typing.get_origin(inner_th) is list:
            inner_type = typing.get_args(inner_th)[0]
            p["type"] = "array"
            p["items"] = {}
            p["items"]["type"] = type2str(inner_type)

            if arg_info.choices:
                p["items"]["enum"] = arg_info.choices
        else:
            p["type"] = type2str(inner_th)
            if arg_info.choices:
                p["enum"] = arg_info.choices
                if (
                    inner_th is int
                    and capabilities
                    and not capabilities.supports_choices_on_int
                ):
                    raise ValueError(
                        f"Parameter '{arg_info.name}' uses 'choices' with 'int', which is not supported by this model."
                    )

        return p

    properties_info: dict[str, dict[str, Any]] = {}
    required_properties: list[str] = []

    for arg_info in fnc_info.arguments.values():
        if arg_info.default is inspect.Parameter.empty:
            required_properties.append(arg_info.name)

        properties_info[arg_info.name] = build_oai_property(arg_info)

    return {
        "type": "function",
        "function": {
            "name": fnc_info.name,
            "description": fnc_info.description,
            "parameters": {
                "type": "object",
                "properties": properties_info,
                "required": required_properties,
            },
        },
    }
