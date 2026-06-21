#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --user -r requirements.txt
python3 -m pip install --user streamlit

if [[ -n "${FRED_API_KEY:-}" && ! -f ".env" ]]; then
  umask 077
  printf 'FRED_API_KEY=%s\n' "${FRED_API_KEY}" > .env
fi

printf 'Packages installed and runtime environment prepared.\n'
