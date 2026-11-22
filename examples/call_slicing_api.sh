#!/usr/bin/env bash
# 调用训练切片规划 API 的示例。用法：
#   ./examples/call_slicing_api.sh [BASE_URL] [REQUEST_JSON]
# 默认 BASE_URL=http://localhost:8000，REQUEST_JSON=examples/slice_request_solved.json
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
REQUEST_JSON="${2:-examples/slice_request_solved.json}"

echo "== GET  /health =="
curl -sS "$BASE_URL/health"
echo
echo "== POST /api/v1/training-slices  ($REQUEST_JSON) =="
curl -sS -X POST "$BASE_URL/api/v1/training-slices" \
  -H 'Content-Type: application/json' \
  -d @"$REQUEST_JSON"
echo
