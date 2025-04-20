python3 -m sglang.launch_server \
--model-path Qwen/Qwen2.5-VL-7B-Instruct-AWQ  \
--chat-template=qwen2-vl \
--mem-fraction-static=0.6 \
--tool-call-parser qwen25 