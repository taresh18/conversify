import logging
from typing import AsyncIterable
import asyncio
import re

from dotenv import load_dotenv
from typing import Union
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import silero, turn_detector
from livekit import rtc
from livekit.agents.metrics import LLMMetrics, TTSMetrics, PipelineEOUMetrics

from custom_plugins.whisper import WhisperSTT
from custom_plugins.kokoro import KokoroTTS
from custom_plugins.openai_llm import OpenaiLLM

load_dotenv(dotenv_path=".env.local")

# Global logger setup - logs written to 'app.log'
logger = logging.getLogger("voice-agent")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,
    filename='app.log',  # Log file for the root logger
    filemode='a'         # Append mode; use 'w' to overwrite each time
)

# Specialized pipeline logger setup - logs written to 'pipeline.log'
pipeline_logger = logging.getLogger('livekit.agents.pipeline')
pipeline_logger.handlers.clear()  # Clear any existing handlers
file_handler = logging.FileHandler('pipeline.log')  # Log file for the pipeline logger
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
pipeline_logger.addHandler(file_handler)


async def clean_text_for_tts(agent: "VoicePipelineAgent", text: Union[str, AsyncIterable[str]]) -> str:
    logger.info(f"before tts cb: {text}")
    def clean(text_chunk: str) -> str:
        # Remove special tags
        cleaned = text_chunk.replace("<think>", "").replace("</think>", "")
        # Remove code blocks enclosed in triple backticks
        cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
        # Remove code blocks enclosed in triple single quotes
        cleaned = re.sub(r"'''(.*?)'''", r'\1', cleaned, flags=re.DOTALL)
        # Remove markdown bold/italic markers
        cleaned = re.sub(r'(\*\*|__)(.*?)\1', r'\2', cleaned)
        cleaned = re.sub(r'(\*|_)(.*?)\1', r'\2', cleaned)
        # Remove inline code markers (backticks)
        cleaned = re.sub(r'`([^`]*)`', r'\1', cleaned)
        # Remove LaTeX inline delimiters: remove one or more backslashes preceding "(" or ")"
        cleaned = re.sub(r'\\+\(', '', cleaned)
        cleaned = re.sub(r'\\+\)', '', cleaned)
        return cleaned

    if isinstance(text, str):
        return clean(text)
    else:
        # For streaming text, collect and concatenate cleaned chunks
        cleaned_chunks = []
        async for chunk in text:
            cleaned_chunks.append(clean(chunk))
        return "".join(cleaned_chunks)


# can be used to warmup the models
def prewarm(proc: JobProcess):
    # load silero weights and store to process userdata
    proc.userdata["vad"] = silero.VAD.load(
                                min_speech_duration=0.20,
                                min_silence_duration=0.50,
                                prefix_padding_duration=0.5,
                                max_buffered_speech=60.0,
                                activation_threshold=0.5,
                                force_cpu=False,
                                sample_rate=16000 
                        )

# for finding latency
end_of_utterance_delay = 0
llm_ttft = 0
tts_ttfb = 0

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text="""You are Conversify, a friendly and efficient assistant integrated into a conversation app that uses multiple components (Voice Activity Detection, Speech-to-Text, Text-to-Speech, and LLM). Your primary task is to process the text provided by the STT component, fully understand the user's query, and deliver short, clear, and satisfying responses.

Tone & Style:
- Answer in a warm, approachable, and friendly manner.
- Keep your responses concise and focused on the query.
- Ensure your language is engaging and simple, suitable for conversion to speech.

Functionality & Behavior:
- Address the user's query accurately, avoiding unnecessary repetition or irrelevant keywords.
- Do not provide information or answers about topics (such as current news headlines) that are outside your knowledge or toolset.
- Avoid repeating preset phrases (for example, refrain from repeating "speak softly" or similar terms).
- If further context is needed to clarify a query, ask briefly without derailing the conversation.

Objective:
- Provide helpful, correct, and satisfying answers that directly support the user's needs, leaving them informed and content with the interaction.
"""
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    stt_obj = WhisperSTT(
        model="deepdml/faster-whisper-large-v3-turbo-ct2",
        device='cuda',
        compute_type='float16'
    )

    llm_obj = OpenaiLLM(
        model="Qwen/Qwen2.5-3B-Instruct-AWQ", 
        # model="abhishekchohan/gemma-3-4b-it-quantized-W4A16", 
        api_key="NULL",
        base_url="http://127.0.0.1:30000/v1",
        temperature="0.4", 
        parallel_tool_calls=False,
        tool_choice = "none",
    )

    tts_obj = KokoroTTS(
        api_key="NULL",
        base_url="http://0.0.0.0:8880/v1",
        model="tts-1"   ,
        voice='af_heart'
    )
    eou = turn_detector.EOUModel(unlikely_threshold=0.0289)

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=stt_obj,
        llm = llm_obj,
        tts = tts_obj,
        # use LiveKit's transformer-based turn detector
        turn_detector=eou, # when not using turn detector with custon stt, it doesnt generate response after first one
        # intial ChatContext with system prompt
        chat_ctx=initial_ctx,
        # whether the agent can be interrupted
        allow_interruptions=True,
        # the minimum speech duration (detected by VAD) required to consider the interruption intentional
        interrupt_speech_duration=0.5,
        # the minimum number of transcribed words needed for the interruption to be considered intentional.
        interrupt_min_words=0,
        # minimal silence duration to consider end of turn
        min_endpointing_delay=0.3,
        # Whether to preemptively synthesize responses
        preemptive_synthesis=True,
        # callback to run before TTS is called, can be used to customize pronounciation
        before_tts_cb=clean_text_for_tts,
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)
        global end_of_utterance_delay
        global llm_ttft
        global tts_ttfb
        if isinstance(agent_metrics, PipelineEOUMetrics):
            end_of_utterance_delay = agent_metrics.end_of_utterance_delay
        elif isinstance(agent_metrics, LLMMetrics):
            llm_ttft = agent_metrics.ttft
        elif isinstance(agent_metrics, TTSMetrics):
            tts_ttfb = agent_metrics.ttfb
            e2e_latency = end_of_utterance_delay + llm_ttft + tts_ttfb
            logger.info(f"TOTAL E2E LATENCY: {e2e_latency}, eou: {end_of_utterance_delay}, ttft: {llm_ttft}, ttfb: {tts_ttfb}")
            # logs
            # TOTAL E2E LATENCY: 2.809550713049248, eou: 1.6684392590541393, ttft: 0.6034322429914027, ttfb: 0.5376792110037059
            # eou: 0.5532792339799926, ttft: 0.6099456179654226, ttfb: 0.4665863789850846 - using gpu for vad
            # TOTAL E2E LATENCY: 1.6311424928717315, eou: 0.5163242398994043, ttft: 0.5858858079882339, ttfb: 0.5289324449840933
            # TOTAL E2E LATENCY: 1.4251687979558483, eou: 0.7649736419552937, ttft: 0.6018818540032953, ttfb: 0.05831330199725926 - kokoro + faster whisper
            # TOTAL E2E LATENCY: 0.7837725100107491, eou: 0.7059646369889379, ttft: 0.02561574208084494, ttfb: 0.05219213094096631 - custom 
            # TOTAL E2E LATENCY: 0.766751405200921, eou: 0.7064768810523674, ttft: 0.02584131306502968, ttfb: 0.03443321108352393 - custom + llm changes to qwen2.5-3b-AWQ

    # Log aggregated summary of usage metrics generated by usage collector
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: ${summary}")
    
    ctx.add_shutdown_callback(log_usage)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"connected to room {ctx.room.name} with participant {participant.identity}")
    agent.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            job_memory_warn_mb=1900,
            load_threshold=1,
            job_memory_limit_mb=5000,
        ),
    )
