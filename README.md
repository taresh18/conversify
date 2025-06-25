# Conversify 🗣️ ✨

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Conversify is a real‑time, low‑latency, voice- and vision-enabled AI assistant built on LiveKit. This project demonstrates highly responsive conversational AI workflows, leveraging locally hosted models.

## Demo Video

[![Watch the demo](assets/thumbnail.jpg)](https://youtu.be/Biva5VGV5Pg)


## ✨ Key Features

- ⚡ **Low Latency**: End-to-end response time under 600 ms.
- 🗣️ **Real‑time Voice**: Natural conversation using local STT and TTS services.
- 🧠 **Local LLM Integration**: Compatible with any OpenAI‑style API (e.g., SGLang, vLLM, Ollama).
- 👀 **Basic Vision**: Processes video frames with multimodal LLM prompts.
- 💾 **Conversational Memory**: Persists context across user sessions.
- 🔧 **Configurable**: All settings managed via `config/config.yaml`.

---

## ⚙️ Prerequisites

- **OS**: Linux or WSL on Windows (tested)
- **Python**: 3.11+
- **Services**:
  - LiveKit Server Cloud (sign up at https://cloud.livekit.io)
  - An LLM inference server with OpenAI-compatible API (e.g., SGLang, vLLM, Ollama)
  - Kokoro FastAPI TTS server (https://github.com/remsky/Kokoro-FastAPI)

---

## 🛠️ Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/taresh18/conversify.git
    cd conversify
    ```

2. **Create a virtual environment** (recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate    # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    python -m conversify.main download-files 
    ```

4. **Configure environment variables**

    ```bash
    cp .env.example .env.local
    nano .env.local  # Add your LiveKit and other credentials
    ```

5. **Update `config/config.yaml`**

    - Set LLM API endpoint and model names
    - Configure STT/TTS server URLs and parameters
    - Adjust vision and memory settings as needed

---

## 🏃 Running the Application

Ensure all external services are running before starting Conversify.

1. **Start the LLM server** (example using provided script)

    ```bash
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

    - Navigate to https://agents-playground.livekit.io
    - Select your LiveKit project and room
    - Join and begin conversation

---

## ⚙️ Configuration

All runtime settings are in `config/config.yaml`. Key options include:

- **STT**: model selection and parameters
- **LLM**: endpoint URLs and model names
- **TTS**: voice options and server settings
- **Vision**: enable/disable frame analysis and thresholds
- **Memory**: persistence and retrieval parameters
- **Logging**: level and file path (`app.log`)

Secrets and credentials reside in `.env.local`, following the template in `.env.example`.

---

## 🏗️ Project Structure

```plaintext
conversify/
├── config/
│   └── config.yaml         # All application settings
├── conversify/
│   ├── core/               # Orchestration and agent logic
│   ├── stt/                # Speech-to-text client
│   ├── tts/                # Text-to-speech client
│   ├── llm/                # LLM integration client
│   ├── livekit/            # LiveKit session & media management
│   └── utils/              # Logger and shared utilities
├── prompts/
│   └── llm.txt             # System prompt for LLM
├── scripts/
│   ├── run_llm.sh
│   ├── run_kokoro.sh
│   └── run_app.sh
├── .env.example            # Template for environment variables
├── .env.local              # Local secrets (ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📚 References

- LiveKit Agents: https://github.com/livekit/agents
- Faster Whisper: https://github.com/SYSTRAN/faster-whisper
- Kokoro FastAPI: https://github.com/remsky/Kokoro-FastAPI
- Memoripy: https://github.com/caspianmoon/memoripy

---

## 📜 License

This project is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

