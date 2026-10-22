#!/bin/bash
# Kill every process whose command line matches the pattern, except this script and the shell that called it (pkill -f from an inline command
# otherwise matches the caller's own command text and kills it; see the exit-144 incidents). Usage: bash killpat.sh '<regex>'
for p in $(pgrep -f "$1"); do [ "$p" != "$$" ] && [ "$p" != "$PPID" ] && kill "$p" 2>/dev/null && echo "killed $p"; done; exit 0
