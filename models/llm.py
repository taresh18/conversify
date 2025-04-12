"""
Language Model implementation for the Conversify system.
"""

import asyncio
import datetime
import logging
import os
import re
import json
from dataclasses import dataclass
from typing import Any, Literal, MutableSet, Union, Optional

import aiohttp
import httpx
import openai
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import Choice

from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    llm,
)
from livekit.agents.llm import (
    LLMCapabilities,
    ToolChoice,
    _create_ai_function_info,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, NotGivenOr, NOT_GIVEN

from core.config import config
from utils import ChatModels

logger = logging.getLogger(__name__)

@dataclass
class OpenaiLLMOptions:
    """Configuration options for OpenaiLLM."""
    model: str | ChatModels
    user: str | None
    temperature: float | None
    parallel_tool_calls: bool | None
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto"
    store: bool | None = None
    metadata: dict[str, str] | None = None
    max_tokens: int | None = None


def _strip_nones(data: dict[str, Any]) -> dict[str, Any]:
    """Remove None values from a dictionary."""
    return {k: v for k, v in data.items() if v is not None}


def truncate_base64_images(messages):
    """Truncate base64 image data for logging purposes.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        List of messages with truncated base64 data
    """
    truncated_messages = []

    for msg in messages:
        content = msg.get('content')

        # If content is a list (e.g., for multimodal input with image)
        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image/") and "base64," in url:
                        # Truncate base64 content
                        new_url = url.split("base64,", 1)[0] + "base64,[TRUNCATED]"
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": new_url}
                        })
                    else:
                        new_content.append(item)
                else:
                    new_content.append(item)
            msg = {**msg, "content": new_content}
        
        # If content is a string, truncate inline base64 data URLs (edge case)
        elif isinstance(content, str):
            truncated = re.sub(
                r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
                lambda m: m.group(0).split("base64,", 1)[0] + "base64,[TRUNCATED]",
                content
            )
            msg = {**msg, "content": truncated}

        truncated_messages.append(msg)

    return truncated_messages


def build_oai_message(msg: llm.ChatMessage, cache_key: Any):
    """Convert a ChatMessage to an OpenAI API message format.
    
    Args:
        msg: The ChatMessage to convert
        cache_key: Cache key for image data
        
    Returns:
        OpenAI API message format
    """
    oai_msg: dict[str, Any] = {"role": msg.role}

    if msg.name:
        oai_msg["name"] = msg.name

    # Add content if provided
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

    # Make sure to provide when function has been called inside the context
    # (+ raw_arguments)
    if msg.tool_calls is not None:
        tool_calls: list[dict[str, Any]] = []
        oai_msg["tool_calls"] = tool_calls
        for fnc in msg.tool_calls:
            tool_calls.append(
                {
                    "id": fnc.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": fnc.function_info.name,
                        "arguments": fnc.raw_arguments,
                    },
                }
            )

    # tool_call_id is set when the message is a response/result to a function call
    # (content is a string in this case)
    if msg.tool_call_id:
        oai_msg["tool_call_id"] = msg.tool_call_id

    return oai_msg


def _build_oai_image_content(image: llm.ChatImage, cache_key: Any):
    """Convert a ChatImage to an OpenAI API image content format.
    
    Args:
        image: The ChatImage to convert
        cache_key: Cache key for image data
        
    Returns:
        OpenAI API image content format
    """
    import base64
    from livekit import rtc
    from livekit.agents import utils

    if isinstance(image.image, str):  # image url
        return {
            "type": "image_url",
            "image_url": {"url": image.image, "detail": image.inference_detail},
        }
    elif isinstance(image.image, rtc.VideoFrame):  # VideoFrame
        if cache_key not in image._cache:
            # Inside our internal implementation, we allow to put extra metadata to
            # each ChatImage (avoid to reencode each time we do a chatcompletion request)
            opts = utils.images.EncodeOptions()
            if image.inference_width and image.inference_height:
                opts.resize_options = utils.images.ResizeOptions(
                    width=image.inference_width,
                    height=image.inference_height,
                    strategy="scale_aspect_fit",
                )

            encoded_data = utils.images.encode(image.image, opts)
            image._cache[cache_key] = base64.b64encode(encoded_data).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image._cache[cache_key]}",
                "detail": image.inference_detail,
            },
        }

    raise ValueError(
        "LiveKit OpenAI Plugin: ChatImage must be an rtc.VideoFrame or a URL"
    )


def build_oai_function_description(
    fnc_info: llm.function_context.FunctionInfo,
    capabilities: llm.LLMCapabilities | None = None,
) -> dict[str, Any]:
    """Convert a FunctionInfo to an OpenAI API function description.
    
    Args:
        fnc_info: The FunctionInfo to convert
        capabilities: LLM capabilities
        
    Returns:
        OpenAI API function description
    """
    import inspect
    import typing
    from livekit.agents.llm.function_context import _is_optional_type

    def build_oai_property(arg_info: llm.function_context.FunctionArgInfo):
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


def _build_oai_context(
    chat_ctx: llm.ChatContext, cache_key: Any
) -> list[ChatCompletionMessageParam]:
    """Convert a ChatContext to a list of OpenAI API messages.
    
    Args:
        chat_ctx: The ChatContext to convert
        cache_key: Cache key for image data
        
    Returns:
        List of OpenAI API messages
    """
    return [build_oai_message(msg, cache_key) for msg in chat_ctx.messages]  # type: ignore


class OpenaiLLM(llm.LLM):
    """LLM implementation using OpenAI API."""
    
    def __init__(
        self,
        *,
        model: Optional[str | ChatModels] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        user: Optional[str] = None,
        client: Optional[openai.AsyncClient] = None,
        temperature: Optional[float] = None,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
        store: Optional[bool] = None,
        metadata: Optional[dict[str, str]] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[httpx.Timeout] = None,
    ) -> None:
        """Initialize the OpenaiLLM instance.
        
        Args:
            model: LLM model to use
            api_key: API key
            base_url: API base URL
            user: User identifier
            client: Optional pre-configured OpenAI AsyncClient
            temperature: Sampling temperature
            parallel_tool_calls: Whether to allow parallel tool calls
            tool_choice: Tool choice strategy
            store: Whether to store the conversation
            metadata: Additional metadata
            max_tokens: Maximum number of tokens to generate
            timeout: HTTP timeout
        """
        super().__init__(
            capabilities=LLMCapabilities(
                supports_choices_on_int=True,
                requires_persistent_functions=False,
            )
        )
        
        # Load from config if not provided
        model = model or config.get('llm.openai.model', 'gpt-4o')
        temperature = temperature if temperature is not None else config.get('llm.openai.temperature', 0.4)
        parallel_tool_calls = parallel_tool_calls if parallel_tool_calls is not None else config.get('llm.openai.parallel_tool_calls', False)
        max_tokens = max_tokens or config.get('llm.openai.max_tokens', 64)
        
        # Use environment variables if not provided
        if api_key is None:
            api_key = config.get_env('OPENAI_API_KEY')
        
        # If base_url is not explicitly set, build it from config    
        if base_url is None:
            api_url = config.get('llm.openai.api_url', 'http://127.0.0.1')
            api_port = config.get('llm.openai.api_port', 30000)
            api_path = config.get('llm.openai.api_path', '/v1')
            
            # Build the complete base URL
            base_url = f"{api_url}:{api_port}{api_path}"
            logger.info(f"Using LLM API URL: {base_url}")
            
            # If base_url from config is empty, try environment variable
            if not base_url:
                base_url = config.get_env('OPENAI_API_BASE_URL')

        self._opts = OpenaiLLMOptions(
            model=model,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            store=store,
            metadata=metadata,
            max_tokens=max_tokens,
        )
        self._client = client or openai.AsyncClient(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=timeout
                if timeout
                else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )
        self._running_fncs: MutableSet[asyncio.Task[Any]] = set()

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: Optional[llm.FunctionContext] = None,
        temperature: Optional[float] = None,
        n: Optional[int] = 1,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Optional[Union[ToolChoice, Literal["auto", "required", "none"]]] = None,
    ) -> "OpenaiLLMStream":
        """Create a chat completion stream.
        
        Args:
            chat_ctx: Chat context
            conn_options: Connection options
            fnc_ctx: Function context
            temperature: Sampling temperature
            n: Number of completions to generate
            parallel_tool_calls: Whether to allow parallel tool calls
            tool_choice: Tool choice strategy
            
        Returns:
            Stream of chat completions
        """
        if parallel_tool_calls is None:
            parallel_tool_calls = self._opts.parallel_tool_calls

        if tool_choice is None:
            tool_choice = self._opts.tool_choice

        if temperature is None:
            temperature = self._opts.temperature

        return OpenaiLLMStream(
            self,
            client=self._client,
            model=self._opts.model,
            user=self._opts.user,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options,
            n=n,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            max_tokens=self._opts.max_tokens,
        )


class OpenaiLLMStream(llm.LLMStream):
    """Stream implementation for OpenaiLLM."""
    
    def __init__(
        self,
        llm_instance: OpenaiLLM,
        *,
        client: openai.AsyncClient,
        model: str | ChatModels,
        user: Optional[str],
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
        fnc_ctx: Optional[llm.FunctionContext],
        temperature: Optional[float],
        n: Optional[int],
        parallel_tool_calls: Optional[bool],
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]],
        max_tokens: Optional[int] = None,
    ) -> None:
        """Initialize the stream.
        
        Args:
            llm_instance: LLM instance
            client: OpenAI AsyncClient
            model: LLM model to use
            user: User identifier
            chat_ctx: Chat context
            conn_options: Connection options
            fnc_ctx: Function context
            temperature: Sampling temperature
            n: Number of completions to generate
            parallel_tool_calls: Whether to allow parallel tool calls
            tool_choice: Tool choice strategy
            max_tokens: Maximum number of tokens to generate
        """
        super().__init__(
            llm_instance, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, conn_options=conn_options
        )
        self._client = client
        self._model = model
        self._llm: OpenaiLLM = llm_instance

        self._user = user
        self._temperature = temperature
        self._n = n
        self._parallel_tool_calls = parallel_tool_calls
        self._tool_choice = tool_choice
        self._max_tokens = max_tokens
        
        # State for tracking function calls
        self._oai_stream: Optional[openai.AsyncStream[ChatCompletionChunk]] = None
        self._tool_call_id: Optional[str] = None
        self._fnc_name: Optional[str] = None
        self._fnc_raw_arguments: Optional[str] = None
        self._tool_index: Optional[int] = None

    async def _run(self) -> None:
        """Run the chat completion stream."""
        if hasattr(self._llm._client, "_refresh_credentials"):
            await self._llm._client._refresh_credentials()

        retryable = True

        try:
            if self._fnc_ctx and len(self._fnc_ctx.ai_functions) > 0:
                tools = [
                    build_oai_function_description(fnc, self._llm._capabilities)
                    for fnc in self._fnc_ctx.ai_functions.values()
                ]
            else:
                tools = None
            
            opts: dict[str, Any] = {
                "tools": tools,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "stream": True,
            }
            
            # Remove None values from the options
            opts = _strip_nones(opts)

            messages = _build_oai_context(self._chat_ctx, id(self))
            truncated_messages = truncate_base64_images(messages)
            logger.info(f"Sending LLM request with truncated_messages: {truncated_messages}")
            
            stream = await self._client.chat.completions.create(
                messages=messages,
                model=self._model,
                **opts,
            )

            async with stream:
                async for chunk in stream:
                    for choice in chunk.choices:
                        chat_chunk = self._parse_choice(chunk.id, choice)
                        if chat_chunk is not None:
                            retryable = False
                            self._event_ch.send_nowait(chat_chunk)

                    if chunk.usage is not None:
                        usage = chunk.usage
                        self._event_ch.send_nowait(
                            llm.ChatChunk(
                                request_id=chunk.id,
                                usage=llm.CompletionUsage(
                                    completion_tokens=usage.completion_tokens,
                                    prompt_tokens=usage.prompt_tokens,
                                    total_tokens=usage.total_tokens,
                                ),
                            )
                        )

        except openai.APITimeoutError:
            raise APITimeoutError(retryable=retryable)
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
            )
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    def _parse_choice(self, id: str, choice: Choice) -> Optional[llm.ChatChunk]:
        """Parse a choice from the API response.
        
        Args:
            id: Request ID
            choice: API response choice
            
        Returns:
            Parsed chat chunk, or None if unable to parse
        """
        delta = choice.delta

        # The delta can be None when using content filtering
        if delta is None:
            return None

        if delta.tool_calls:
            # Check if we have functions to calls
            for tool in delta.tool_calls:
                if not tool.function:
                    continue  # OpenAI may add other tools in the future

                call_chunk = None
                if self._tool_call_id and tool.id and tool.index != self._tool_index:
                    call_chunk = self._try_build_function(id, choice)

                if tool.function.name:
                    self._tool_index = tool.index
                    self._tool_call_id = tool.id
                    self._fnc_name = tool.function.name
                    self._fnc_raw_arguments = tool.function.arguments or ""
                elif tool.function.arguments:
                    self._fnc_raw_arguments += tool.function.arguments  # type: ignore

                if call_chunk is not None:
                    return call_chunk

        if choice.finish_reason in ("tool_calls", "stop") and self._tool_call_id:
            # We're done with the tool calls, run the last one
            return self._try_build_function(id, choice)

        return llm.ChatChunk(
            request_id=id,
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(content=delta.content, role="assistant"),
                    index=choice.index,
                )
            ],
        )

    def _try_build_function(self, id: str, choice: Choice) -> Optional[llm.ChatChunk]:
        """Try to build a function call.
        
        Args:
            id: Request ID
            choice: API response choice
            
        Returns:
            Chat chunk with function call, or None if unable to build
        """
        if not self._fnc_ctx:
            logger.warning("OpenAI stream tried to run function without function context")
            return None

        if self._tool_call_id is None:
            logger.warning(
                "OpenAI stream tried to run function but tool_call_id is not set"
            )
            return None

        if self._fnc_name is None or self._fnc_raw_arguments is None:
            logger.warning(
                "OpenAI stream tried to call a function but raw_arguments and fnc_name are not set"
            )
            return None

        fnc_info = _create_ai_function_info(
            self._fnc_ctx, self._tool_call_id, self._fnc_name, self._fnc_raw_arguments
        )

        self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None
        self._function_calls_info.append(fnc_info)

        return llm.ChatChunk(
            request_id=id,
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        tool_calls=[fnc_info],
                        content=choice.delta.content,
                    ),
                    index=choice.index,
                )
            ],
        ) 