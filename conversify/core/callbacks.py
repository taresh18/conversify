import logging
import asyncio
from typing import Dict, Any

from livekit.agents import AgentSession, metrics
from livekit.agents.voice import MetricsCollectedEvent
from livekit.agents.metrics import LLMMetrics, TTSMetrics, EOUMetrics

from .agent import ConversifyAgent

logger = logging.getLogger(__name__)

# Globals for metrics callback
end_of_utterance_delay = 0
llm_ttft = 0
tts_ttfb = 0
usage_collector = metrics.UsageCollector()


def metrics_callback(session: AgentSession):
    """Sets up the callback for collecting and logging session metrics."""
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        agent_metrics = ev.metrics
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)
        
        # Access globals safely
        global end_of_utterance_delay, llm_ttft, tts_ttfb
        
        if isinstance(agent_metrics, EOUMetrics):
            end_of_utterance_delay = agent_metrics.end_of_utterance_delay
        elif isinstance(agent_metrics, LLMMetrics):
            llm_ttft = agent_metrics.ttft
        elif isinstance(agent_metrics, TTSMetrics):
            tts_ttfb = agent_metrics.ttfb
            # Calculate E2E latency only when TTS metrics arrive (last step)
            e2e_latency = end_of_utterance_delay + llm_ttft + tts_ttfb
            logger.info(f"TOTAL END TO END LATENCY --> {e2e_latency:.3f}s, EOU: {end_of_utterance_delay:.3f}s, LLM: {llm_ttft:.3f}s, TTS: {tts_ttfb:.3f}s")
            # Reset for next interaction cycle
            end_of_utterance_delay = 0
            llm_ttft = 0
            tts_ttfb = 0


async def shutdown_callback(agent: ConversifyAgent, video_task: asyncio.Task | None):
    """Handles graceful shutdown logic: cancels tasks, logs usage, saves memory."""
    logger.info("Application shutdown initiated")
    
    # Cancel video task
    if video_task and not video_task.done():
        logger.info("Attempting to cancel video processing task...")
        try:
            video_task.cancel()
            await video_task 
            logger.info("Video processing task successfully cancelled.")
        except asyncio.CancelledError:
            logger.info("Video processing task was already cancelled or finished.")
        except Exception as e:
            logger.error(f"Error during video task cancellation: {e}", exc_info=True)

    # Log usage summary
    summary: Dict[str, Any] = usage_collector.get_summary()
    logger.info(f"Usage Summary: {summary}")

    # Save conversation history if available
    if agent.memory_handler:
        try:
            logger.info("Saving conversation memory...")
            if not hasattr(agent, 'chat_ctx') or agent.chat_ctx is None:
                logger.warning("Agent chat context is not available, skipping memory save.")
            else:
                await agent.memory_handler.save_memory(agent.chat_ctx)
                logger.info("Conversation memory saved.")
        except Exception as e:
            logger.error(f"Error saving conversation memory: {e}", exc_info=True)
    else:
        logger.info("Memory handler not available, skipping memory save.")
    
    logger.info("Shutdown callback finished.") 