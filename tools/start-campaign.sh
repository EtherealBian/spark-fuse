#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh
exec "$PY" -u tools/campaign.py "$@"
