#!/usr/bin/env bash
# Shared runtime-path parsing for portable batch-launch scripts.

parse_batch_runtime_args() {
  KVREUSE_PYTHON="${KVREUSE_PYTHON:-python}"
  RELAY_PYTHON="${RELAY_PYTHON:-}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --kvreuse-python) KVREUSE_PYTHON="${2:-}"; shift 2 ;;
      --relay-python) RELAY_PYTHON="${2:-}"; shift 2 ;;
      -h|--help)
        echo "Usage: $0 [--kvreuse-python PATH] [--relay-python PATH]" >&2
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        echo "Usage: $0 [--kvreuse-python PATH] [--relay-python PATH]" >&2
        exit 2
        ;;
    esac
  done
  [[ -n "$KVREUSE_PYTHON" ]] || { echo "--kvreuse-python requires a path" >&2; exit 2; }
  RUNTIME_ARGS=(--kvreuse-python "$KVREUSE_PYTHON")
  if [[ -n "$RELAY_PYTHON" ]]; then
    RUNTIME_ARGS+=(--relay-python "$RELAY_PYTHON")
  fi
}
