"""
Main entry point for the Conversify system.
"""

import os
import sys

# Set up logging first, before other imports
from core.config import config
from core.logging import setup_logging

# Initialize logging
logger = setup_logging()
logger.info("Logging system initialized")

# Now import the rest of the components
from livekit.agents import cli, WorkerOptions
from core.agent import ConversifyAgent

def main():
    """Main entry point for the application."""
    
    # Create agent instance
    agent = ConversifyAgent()
    
    # Run the agent
    logger.info("Starting Conversify agent")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=agent.entrypoint,
            prewarm_fnc=agent.prewarm,
            job_memory_warn_mb=config.get('worker.job_memory_warn_mb', 1900),
            load_threshold=config.get('worker.load_threshold', 1),
            job_memory_limit_mb=config.get('worker.job_memory_limit_mb', 10000),
        ),
    )

if __name__ == "__main__":
    main() 