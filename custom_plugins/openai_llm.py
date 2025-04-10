# Copyright 2023 LiveKit, Inc.
#

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import datetime
import os
from dataclasses import dataclass
from typing import Any, Literal, MutableSet, Union

import aiohttp
import httpx
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
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

import openai
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import Choice

from .utils import ChatModels, build_oai_message, build_oai_function_description

import logging
logger = logging.getLogger(__name__)


@dataclass
class LLMOptions:
    model: str | ChatModels
    user: str | None
    temperature: float | None
    parallel_tool_calls: bool | None
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto"
    store: bool | None = None
    metadata: dict[str, str] | None = None
    max_tokens: int | None = None


class OpenaiLLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        user: str | None = None,
        client: openai.AsyncClient | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
        store: bool | None = None,
        metadata: dict[str, str] | None = None,
        max_tokens: int | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """
        Create a new instance of OpenAI LLM.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.
        """
        super().__init__(
            capabilities=LLMCapabilities(
                supports_choices_on_int=True,
                requires_persistent_functions=False,
            )
        )

        self._opts = LLMOptions(
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
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]]
        | None = None,
    ) -> "LLMStream":
        if parallel_tool_calls is None:
            parallel_tool_calls = self._opts.parallel_tool_calls

        if tool_choice is None:
            tool_choice = self._opts.tool_choice

        if temperature is None:
            temperature = self._opts.temperature

        return LLMStream(
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
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        client: openai.AsyncClient,
        model: str | ChatModels,
        user: str | None,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
        fnc_ctx: llm.FunctionContext | None,
        temperature: float | None,
        n: int | None,
        parallel_tool_calls: bool | None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]],
    ) -> None:
        super().__init__(
            llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, conn_options=conn_options
        )
        self._client = client
        self._model = model
        self._llm: LLM = llm

        self._user = user
        self._temperature = temperature
        self._n = n
        self._parallel_tool_calls = parallel_tool_calls
        self._tool_choice = tool_choice

    async def _run(self) -> None:
        if hasattr(self._llm._client, "_refresh_credentials"):
            await self._llm._client._refresh_credentials()

        # current function call that we're waiting for full completion (args are streamed)
        # (defined inside the _run method to make sure the state is reset for each run/attempt)
        self._oai_stream: openai.AsyncStream[ChatCompletionChunk] | None = None
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None
        self._tool_index: int | None = None
        retryable = True

        try:
            if self._fnc_ctx and len(self._fnc_ctx.ai_functions) > 0:
                tools = [
                    build_oai_function_description(fnc, self._llm._capabilities)
                    for fnc in self._fnc_ctx.ai_functions.values()
                ]
            else:
                tools = None

            tools = None

            opts: dict[str, Any] = {
                "tools": tools,
                "parallel_tool_calls": self._parallel_tool_calls if tools else None,
                "tool_choice": (
                    {"type": "function", "function": {"name": self._tool_choice.name}}
                    if isinstance(self._tool_choice, ToolChoice)
                    else self._tool_choice
                )
                if tools is not None
                else None,
                "tools": [],
                "temperature": self._temperature,
                "metadata": self._llm._opts.metadata,
                "max_tokens": self._llm._opts.max_tokens,
                "store": self._llm._opts.store,
                "n": self._n,
                "stream": True,
                "stream_options": {"include_usage": True},
                "user": self._user or openai.NOT_GIVEN,
            }
            # remove None values from the options
            opts = _strip_nones(opts)

            messages = _build_oai_context(self._chat_ctx, id(self))

            stream = await self._client.chat.completions.create(
                messages=messages,
                model=self._model,
                # **opts,
                temperature=0.4,
                max_tokens=64,
                stream=True
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

    def _parse_choice(self, id: str, choice: Choice) -> llm.ChatChunk | None:
        delta = choice.delta

        # https://github.com/livekit/agents/issues/688
        # the delta can be None when using Azure OpenAI using content filtering
        if delta is None:
            return None

        if delta.tool_calls:
            # check if we have functions to calls
            for tool in delta.tool_calls:
                if not tool.function:
                    continue  # oai may add other tools in the future

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
            # we're done with the tool calls, run the last one
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

    def _try_build_function(self, id: str, choice: Choice) -> llm.ChatChunk | None:
        if not self._fnc_ctx:
            logger.warning("oai stream tried to run function without function context")
            return None

        if self._tool_call_id is None:
            logger.warning(
                "oai stream tried to run function but tool_call_id is not set"
            )
            return None

        if self._fnc_name is None or self._fnc_raw_arguments is None:
            logger.warning(
                "oai stream tried to call a function but raw_arguments and fnc_name are not set"
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


def _build_oai_context(
    chat_ctx: llm.ChatContext, cache_key: Any
) -> list[ChatCompletionMessageParam]:
    return [build_oai_message(msg, cache_key) for msg in chat_ctx.messages]  # type: ignore


def _strip_nones(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


def _get_api_key(env_var: str, key: str | None) -> str:
    key = key or os.environ.get(env_var)
    if not key:
        raise ValueError(
            f"{env_var} is required, either as argument or set {env_var} environmental variable"
        )
    return key
