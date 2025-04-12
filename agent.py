import logging
from typing import AsyncIterable
import asyncio
import re
from typing import Annotated
import types  # Add import for types module


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
from livekit import agents

from custom_plugins.whisper import WhisperSTT
from custom_plugins.kokoro import KokoroTTS
from custom_plugins.openai_llm import OpenaiLLM
from PIL import Image
from livekit.plugins import google
from livekit.agents.llm import ChatContext, ChatImage, ChatMessage


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
file_handler = logging.FileHandler('app.log')  # Log file for the pipeline logger
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
    

class VisionFunctions(agents.llm.FunctionContext):
    @agents.llm.ai_callable(description="Call this when the user asks a question that requires analyzing the current visual input or webcam feed.")
    async def analyze_current_view(
        self,
        user_query: Annotated[str, agents.llm.TypeInfo(description="The user's specific question about the visual input.")]
    ):
        # This function itself might not do much,
        # the real work happens in the 'function_calls_finished' handler.
        # It signals that vision is needed.
        logger.info(f"Vision function called for query: {user_query}")
        # We'll handle the actual image sending in the event handler
        return {"status": "Acknowledged. Will analyze the view."}


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
        text="""You are Conversify, a friendly and efficient assistant integrated into a conversation app that uses multiple components (Voice Activity Detection, Speech-to-Text, Text-to-Speech, and LLM). Your primary task is to process the text provided by the STT component, fully understand the user's query, and deliver short, clear, and satisfying responses. You also have access to visual input.

Tone & Style:
- Answer in a warm, approachable, and friendly manner.
- Keep your responses concise and focused on the query.
- Ensure your language is engaging and simple, suitable for conversion to speech.

Functionality & Behavior:
- Address the user's query accurately, avoiding unnecessary repetition or irrelevant keywords.
- **Vision Tool Usage:** If the user asks a question that requires understanding the current visual input or webcam feed (e.g., "What do you see?", "Describe this object", "What color is the item I'm holding?", "Is there anything on the desk?"), you **MUST** call the `analyze_current_view` function. Pass the user's specific question as the `user_query` argument to this function. Do **not** attempt to answer questions about visual input directly without using this function first.
- Do not provide information or answers about topics (such as current news headlines) that are outside your knowledge or toolset, unless a specific tool provides that information.
- Avoid repeating preset phrases (for example, refrain from repeating "speak softly" or similar terms).
- If further context is needed to clarify a query, ask briefly without derailing the conversation.

Objective:
- Provide helpful, correct, and satisfying answers that directly support the user's needs, leveraging your text understanding, visual input (via the tool), and any other available tools, leaving the user informed and content with the interaction."""
    )
    
    # Create a new chat context with a proper system message
    chat_context = ChatContext()
    # Add the system message
    chat_context.messages.append(
        ChatMessage(
            role="system",
            content=(
                "Your name is Conversify. You are a funny, witty bot. Your interface with users will be voice and vision."
                "Respond with short and concise answers. Avoid using unpronouncable punctuation or emojis."
            ),
        )
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    stt_obj = WhisperSTT(
        model="deepdml/faster-whisper-large-v3-turbo-ct2",
        device='cuda',
        compute_type='float16'
    )

    llm_obj = OpenaiLLM(
        # model="Qwen/Qwen2.5-VL-3B-Instruct-AWQ", 
        model="abhishekchohan/gemma-3-4b-it-quantized-W4A16", 
        api_key="NULL",
        base_url="http://127.0.0.1:30000/v1",
        temperature="0.4", 
        parallel_tool_calls=False,
        # tool_choice = "none",
    )
    
    # llm_obj = google.LLM(
    #     model="gemini-2.0-flash-exp", 
    #     # model="abhishekchohan/gemma-3-4b-it-quantized-W4A16", 
    #     # api_key="NULL",
    #     # base_url="http://127.0.0.1:30000/v1",
    #     # temperature="0.4", 
    #     # parallel_tool_calls=False,
    #     # # tool_choice = "none",
    # )

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
        # chat_ctx=initial_ctx,
        chat_ctx=chat_context,
        # whether the agent can be interrupted
        allow_interruptions=True,
        # the minimum speech duration (detected by VAD) required to consider the interruption intentional
        interrupt_speech_duration=0.5,
        
        fnc_ctx=VisionFunctions(),
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
            
    latest_image: rtc.VideoFrame | None = None # Variable to store the latest frame
    
    async def video_processing_loop():
        nonlocal latest_image
        print("Looking for video track...")
        # Wait for a remote participant with a video track
        video_track = None
        while video_track is None:
            for participant in ctx.room.remote_participants.values():
                for track_pub in participant.track_publications.values():
                    if track_pub.kind == rtc.TrackKind.KIND_VIDEO and track_pub.track:
                        # Found a video track, subscribe if not already
                        if not track_pub.subscribed:
                            track_pub.set_subscribed(True)
                        # Wait a moment for subscription to complete if needed
                        await asyncio.sleep(0.5)
                        if isinstance(track_pub.track, rtc.RemoteVideoTrack):
                            video_track = track_pub.track
                            print(f"Using video track: {video_track.sid} from {participant.identity}")
                            break
                if video_track:
                    break
            if video_track is None:
                await asyncio.sleep(1) # Wait and retry if no track found yet

        # Process the video stream
        video_stream = rtc.VideoStream(video_track)
        try:
            async for event in video_stream:
                latest_image = event.frame
                # Optional: Add a small sleep if processing every single frame is too much
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error processing video stream: {e}")
        finally:
            print("Video stream ended.")
            await video_stream.aclose() # Ensure stream is closed

    # Start the video processing loop as a background task
    video_task = asyncio.create_task(video_processing_loop())
    
    
    async def _vision_call(called_functions):
        nonlocal latest_image # Make sure latest_image is accessible
        nonlocal llm_obj      # Make sure your LLM object is accessible
        nonlocal agent        # Make sure the agent is accessible
        
        logger.info(f"called_functions: {called_functions}")
        # logger.info(f"latest_image: {latest_image}")
        for func_call in called_functions:
            logger.info(f"func_call: {func_call}")
            if func_call.call_info.function_info.name == "analyze_current_view":
                user_query = func_call.call_info.function_info.arguments.get("user_query", "Describe what you see.")

                if latest_image is None:
                    await agent.say("Sorry, I don't have a visual input right now.", allow_interruptions=True)
                    return
                
                content: list[str | ChatImage] = [user_query]
                content.append(ChatImage(image=latest_image))
                
                logger.info(f"chat_context.messages: {chat_context.messages}")
                
                # Add the new user message with image
                chat_context.messages.append(ChatMessage(role="user", content=content))
                
                logger.info(f"updated chat_context.messages: {chat_context.messages}")
                stream = llm_obj.chat(chat_ctx=chat_context)
                await agent.say(stream, allow_interruptions=True)
        
            
    @agent.on("function_calls_finished")
    def handle_vision_call(called_functions: list[agents.llm.CalledFunction]):
        if len(called_functions) == 0:
            return           
        asyncio.create_task(_vision_call(called_functions))
        
            
    # Ensure the task is cancelled on shutdown
    def cancel_video_task():
        if video_task and not video_task.done():
            video_task.cancel()
            print("Video processing task cancelled.")

    ctx.add_shutdown_callback(cancel_video_task)
        

    # # Log aggregated summary of usage metrics generated by usage collector
    # async def log_usage():
    #     summary = usage_collector.get_summary()
    #     logger.info(f"Usage: ${summary}")
    
    # ctx.add_shutdown_callback(log_usage)

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
            job_memory_limit_mb=10000,
        ),
    )
