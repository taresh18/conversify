# Conversify ✨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Conversify is a real‑time, low‑latency, voice- and vision-enabled AI assistant built on LiveKit. This project demonstrates highly responsive conversational AI workflows, leveraging locally hosted models.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Running the Application](#running-the-application)
5. [Configuration](#configuration)
6. [Project Structure](#project-structure)
7. [TODO](#todo)
8. [References](#references)
9. [License](#license)

---

## Key Features

- ⚡ **Low Latency**: End-to-end response time under 600 ms.
- 🗣️ **Real‑time Voice**: Natural conversation using local STT and TTS services.
- 🧠 **Local LLM Integration**: Compatible with any OpenAI‑style API (e.g., SGLang, vLLM, Ollama).
- 👀 **Basic Vision**: Processes video frames with multimodal LLM prompts.
- 💾 **Conversational Memory**: Persists context across user sessions.
- 🔧 **Configurable**: All settings managed via `config/config.yaml`.

---

## Prerequisites

- **OS**: Linux or WSL on Windows (tested)
- **Python**: 3.11+
- **Services**:
  - LiveKit Server Cloud - [sign up](https://cloud.livekit.io/login) and create a project
  - An LLM inference server with OpenAI-compatible API (e.g., SGLang, vLLM, Ollama)
  - Kokoro FastAPI TTS server

---

## Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/taresh18/conversify-speech.git
    cd conversify-speech
    ```

2. **Create a virtual environment** (recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate    # Linux/macOS
    # venv\Scripts\activate     # Windows
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure environment variables**

    ```bash
    cp .env.example .env.local
    nano .env.local  # Add your LiveKit server and project credentials
    ```

5. **Update `config/config.yaml`**

    - Set LLM API endpoint and model names
    - Configure STT/TTS server URLs and parameters
    - Adjust vision and memory settings as needed

---

## Running the Application

1. **Start the LLM server**

    ```bash
    # this starts a sglang server
    chmod +x ./scripts/run_llm.sh
    ./scripts/run_llm.sh &
    ```

2. **Start the Kokoro TTS server**

    ```bash
    chmod +x ./scripts/run_kokoro.sh
    ./scripts/run_kokoro.sh &
    ```

3. **Launch Conversify**

    ```bash
    chmod +x ./scripts/run_app.sh
    ./scripts/run_app.sh
    ```

4. **Interact via LiveKit Agents Playground**

    - Navigate to [agents-playground](https://agents-playground.livekit.io)
    - Select your project
    - Connect to the room and begin conversation

---

## Configuration

All runtime settings are in `config/config.yaml`. Key options include:

- **STT**: model selection and parameters
- **LLM**: endpoint URLs and model names
- **TTS**: voice options and server settings
- **Vision**: enable/disable frame analysis and thresholds
- **Memory**: persistence and retrieval parameters
- **Logging**: level and file path (`app.log`)

Secrets and credentials reside in `.env.local`, following the template in `.env.example`.

---

## Project Structure

```plaintext
conversify-speech/
├── conversify/
│   ├── core/               # Core agent logic, vision, memory
│   ├── data/               # Local memory store and model cache files
│   ├── models/             # Interfaces for interfaces for STT, TTS, LLM
│   ├── prompts/            # System prompts
│   ├── utils/              # Utilities 
│   └── main.py             # Main application entry point
├── scripts/
│   ├── run_llm.sh
│   ├── run_kokoro.sh
│   └── run_app.sh
├── .env.example            # Template for environment variables
├── .env.local              # Local secrets (ignored)
├── config.yaml             # All application settings 
├── requirements.txt
├── .gitignore
└── README.md
```

---

## TODO

- Enhance vision-triggered actions and robustness
- Optimize memory retrieval strategies
- Support alternative TTS engines (e.g., Orpheus, Sesame-CSM)
- Tool calling

---

## References

- LiveKit Agents: https://github.com/livekit/agents
- Faster Whisper: https://github.com/SYSTRAN/faster-whisper
- Kokoro FastAPI: https://github.com/remsky/Kokoro-FastAPI
- Memoripy: https://github.com/caspianmoon/memoripy

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

