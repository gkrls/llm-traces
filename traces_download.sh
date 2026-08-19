
#!/usr/bin/env bash

get() {
  local file="traces/${2:-$(basename "${1%%\?*}")}"
  [ -f "$file" ] && { echo "have $file"; return; }
  echo "get  $file"
  curl -fL --retry 3 --create-dirs -o "$file" "$1" || rm -f "$file"
}

# Mooncake -- prefix ids at 512 tokens
get https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl mooncake_conversation.jsonl
get https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl mooncake_toolagent.jsonl
get https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl mooncake_synthetic.jsonl

# ShareGPT -- raw text, ~630 MB
get https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json sharegpt.json

# Qwen-Bailian -- prefix ids at 16 tokens
get https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/raw/refs/heads/main/qwen_coder_blksz_16.jsonl qwen_bailian_coder.jsonl
get https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/raw/refs/heads/main/qwen_thinking_blksz_16.jsonl qwen_bailian_think.jsonl
get https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/raw/refs/heads/main/qwen_traceA_blksz_16.jsonl qwen_bailian_chat.jsonl
get https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/raw/refs/heads/main/qwen_traceB_blksz_16.jsonl qwen_bailian_api_task.jsonl

echo
ls -lh traces
