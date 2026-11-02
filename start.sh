#!/usr/bin/env bash
# One command on a clean machine:  ./start.sh      (needs python3 >= 3.9 with venv + pip)
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
.venv/bin/pip install -q -r requirements.txt
exec .venv/bin/python run.py "$@"
