#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export STREAM_PORT=${STREAM_PORT:-18080}
# Absolute uv path (2026-05-14): eliminates PATH dependency when invoked
# from systemd (--user unit doesn't inherit ~/.local/bin via default PATH).
exec /home/hafnium/.local/bin/uv run fastmcp run mujoco_mcp_server.py --transport sse --host 0.0.0.0 --port 9200
