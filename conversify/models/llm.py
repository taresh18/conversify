from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import httpx

import openai
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, llm
from livekit.agents.llm import ToolChoice, utils as llm_utils
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import FunctionTool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionToolChoiceOptionParam,
    completion_create_params,
)
from openai.types.chat.chat_completion_chunk import Choice

from .utils import to_chat_ctx, to_fnc_ctx


@dataclass
class _LLMOptions:
    model: str
    temperature: NotGivenOr[float]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice]


class OpenaiLLM(llm.LLM):
    def __init__(
        self,
        config: Dict[str, Any],
        client: openai.AsyncClient | None = None,
    ) -> None:
        """
        Create a new instance of OpenAI LLM using configuration.

        Args:
            client: Pre-configured OpenAI client
            config: Configuration dictionary (from config.yaml)
        """
        super().__init__()
        
        llm_config = config['llm']
        
        model = llm_config['model']
        api_key = llm_config['api_key']
        base_url = llm_config['base_url']
        temperature = llm_config['temperature']
        parallel_tool_calls = llm_config['parallel_tool_calls']
        tool_choice = llm_config['tool_choice']
        
        timeout = httpx.Timeout(
            connect=15.0,
            read=5.0,
            write=5.0,
            pool=5.0
        )
        
        self._opts = _LLMOptions(
            model=model,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )
        
        # Use the provided client or create a new one with configured settings
        self._client = client or openai.AsyncClient(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )
  
    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[
            completion_create_params.ResponseFormat | type[llm_utils.ResponseFormatT]
        ] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        extra = {}
        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        parallel_tool_calls = (
            parallel_tool_calls if is_given(parallel_tool_calls) else self._opts.parallel_tool_calls
        )
        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice
        if is_given(tool_choice):
            oai_tool_choice: ChatCompletionToolChoiceOptionParam
            if isinstance(tool_choice, dict):
                oai_tool_choice = {
                    "type": "function",
                    "function": {"name": tool_choice["function"]["name"]},
                }
                extra["tool_choice"] = oai_tool_choice
            elif tool_choice in ("auto", "required", "none"):
                oai_tool_choice = tool_choice
                extra["tool_choice"] = oai_tool_choice

        if is_given(response_format):
            extra["response_format"] = llm_utils.to_openai_response_format(response_format)

        return LLMStream(
            self,
            model=self._opts.model,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        model: str,
        client: openai.AsyncClient,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._client = client
        self._llm = llm
        self._extra_kwargs = extra_kwargs

    async def _run(self) -> None:
        # current function call that we're waiting for full completion (args are streamed)
        # (defined inside the _run method to make sure the state is reset for each run/attempt)
        self._oai_stream: openai.AsyncStream[ChatCompletionChunk] | None = None
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None
        self._tool_index: int | None = None
        retryable = True

        try:
            self._oai_stream = stream = await self._client.chat.completions.create(
                messages=to_chat_ctx(self._chat_ctx, id(self._llm)),
                tools=to_fnc_ctx(self._tools) if self._tools else openai.NOT_GIVEN,
                model=self._model,
                stream_options={"include_usage": True},
                stream=True,
                **self._extra_kwargs,
            )

            async with stream:
                async for chunk in stream:
                    for choice in chunk.choices:
                        chat_chunk = self._parse_choice(chunk.id, choice)
                        if chat_chunk is not None:
                            retryable = False
                            self._event_ch.send_nowait(chat_chunk)

                    if chunk.usage is not None:
                        retryable = False
                        tokens_details = chunk.usage.prompt_tokens_details
                        cached_tokens = tokens_details.cached_tokens if tokens_details else 0
                        chunk = llm.ChatChunk(
                            id=chunk.id,
                            usage=llm.CompletionUsage(
                                completion_tokens=chunk.usage.completion_tokens,
                                prompt_tokens=chunk.usage.prompt_tokens,
                                prompt_cached_tokens=cached_tokens or 0,
                                total_tokens=chunk.usage.total_tokens,
                            ),
                        )
                        self._event_ch.send_nowait(chunk)

        except openai.APITimeoutError:
            raise APITimeoutError(retryable=retryable) from None
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
                retryable=retryable,
            ) from None
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    def _parse_choice(self, id: str, choice: Choice) -> llm.ChatChunk | None:
        delta = choice.delta

        # https://github.com/livekit/agents/issues/688
        # the delta can be None when using Azure OpenAI (content filtering)
        if delta is None:
            return None

        if delta.tool_calls:
            for tool in delta.tool_calls:
                if not tool.function:
                    continue

                call_chunk = None
                if self._tool_call_id and tool.id and tool.index != self._tool_index:
                    call_chunk = llm.ChatChunk(
                        id=id,
                        delta=llm.ChoiceDelta(
                            role="assistant",
                            content=delta.content,
                            tool_calls=[
                                llm.FunctionToolCall(
                                    arguments=self._fnc_raw_arguments or "",
                                    name=self._fnc_name or "",
                                    call_id=self._tool_call_id or "",
                                )
                            ],
                        ),
                    )
                    self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None
                    self._tool_index = None
                    return call_chunk

                if tool.id and not self._tool_call_id:
                    self._tool_call_id = tool.id
                    self._tool_index = tool.index

                if tool.function.name and not self._fnc_name:
                    self._fnc_name = tool.function.name

                if tool.function.arguments:
                    current = self._fnc_raw_arguments or ""
                    self._fnc_raw_arguments = current + tool.function.arguments

                return None

            return None

        if delta.content == "":
            return None

        return llm.ChatChunk(
            id=id,
            delta=llm.ChoiceDelta(
                role="assistant",
                content=delta.content,
            ),
        )
