
#!/usr/bin/env bash

get() {
  local file="traces/${2:-$(basename "${1%%\?*}")}"
  [ -f "$file" ] && { echo "have $file"; return; }
  echo "get  $file"
  curl -fL --retry 3 --create-dirs -o "$file" "$1" || rm -f "$file"
}

# Mooncake -- prefix ids at 512 tokens
get https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl #mooncake_conversation.jsonl
get https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl #mooncake_toolagent.jsonl
get https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl #mooncake_synthetic.jsonl

# ShareGPT -- raw text, ~630 MB
get https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json #sharegpt.json

# Qwen-Bailian -- prefix ids at 16 tokens
get https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/raw/refs/heads/main/qwen_coder_blksz_16.jsonl #qwen_bailian_coder.jsonl
get https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/raw/refs/heads/main/qwen_thinking_blksz_16.jsonl #qwen_bailian_think.jsonl
get https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/raw/refs/heads/main/qwen_traceA_blksz_16.jsonl #qwen_bailian_chat.jsonl
get https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/raw/refs/heads/main/qwen_traceB_blksz_16.jsonl #qwen_bailian_api_task.jsonl


# WildChat-1M -- raw text WITH real timestamps, 14 parquet shards, ~3.4 GB.
# Not gated. Drop shards you do not want; wildchat.py reads whatever is present,
# or takes a count:  python wildchat.py 3
WC=https://huggingface.co/datasets/allenai/WildChat-1M/resolve/main/data
# there are 13 shards in total
for i in $(seq -w 0 6); do
  get $WC/train-000$i-of-00014.parquet wildchat-$i.parquet
done

# SWE-rebench OpenHands -- 67k agent trajectories, ~64 turns each, one file.
# No timestamps; swerebench.py interleaves by turn like sharegpt.py does.
get https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/resolve/main/trajectories.parquet swe-rebench.parquet

echo
ls -lh traces
