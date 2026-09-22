#!/usr/bin/env bash
# Run the studio with the local DeepSeek credentials.
#
# Only the subcommands that talk to a model or need asset roots get --source.
# 'render' executes an already-authored plan and must stay credential-free.
set -euo pipefail
export LLM_API_KEY="$(tr -d '\r\n' < /mnt/c/Users/GMH13/Desktop/dsapikey.txt)"
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_MODEL="deepseek-flash"
export LLM_REASONING_EFFORT="high"
JAR="/mnt/c/Users/GMH13/Release 2.8.3/.minecraft/versions/1.12.2-Forge_14.23.5.28641/1.12.2-Forge_14.23.5.28641.jar"
cd /home/gmh/mc-art
case "${1:-}" in
  render)
    exec python3 -m studio_next "$@"
    ;;
  *)
    exec python3 -m studio_next "$@" --source "$JAR"
    ;;
esac
