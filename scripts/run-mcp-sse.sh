#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-egl}
export STREAM_PORT=${STREAM_PORT:-18080}
exec uv run fastmcp run mujoco_mcp_server.py --transport sse --host 0.0.0.0 --port 9200
