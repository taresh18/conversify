# Conversify

A low-latency voice and vision-enabled AI assistant built with LiveKit.

## Overview

Conversify is a highly responsive AI assistant that processes speech, understands visual input, and generates natural spoken responses with remarkably low latency. It integrates multiple AI components into a seamless conversation system.

## Key Features

- **Low Latency Processing**: Complete end-to-end processing in under 600ms
- **Voice Input/Output**: Natural speech interaction
- **Visual Understanding**: Can process and respond to visual information
- **Conversational Memory**: Retain context from past conversations

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU 
- LiveKit server

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/conversify.git
   cd conversify
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   # Create a local environment file from example
   cp conversify/.env.example
   # Edit with your settings
   nano conversify/.env.local
   ```

## Running the Application

```bash
cd conversify
python main.py prod
```

You can interact with application using the livekit agents playground [livekit agents playground](https://agents-playground.livekit.io/) 

## Configuration

All settings are managed through:
- `.env.local` file for API keys and endpoints
- `config.yaml` for component settings and parameters

Key configuration options:
- STT model selection and parameters
- LLM endpoints and model selection
- TTS voice and server settings
- Video processing toggle
- Voice activity detection parameters

## TODO

- Improve vision capabilities
- Imporve memory feature
- Integrate Orpheus TTS for enhanced voice quality
- Tool calling support ?

## References

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Kokoro FastAPI](https://github.com/remsky/Kokoro-FastAPI) 
- [Memoripy](https://github.com/caspianmoon/memoripy) 
