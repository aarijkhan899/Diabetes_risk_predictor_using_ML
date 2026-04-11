#!/bin/bash
set -e
cd /app
rm -f tmp/pids/server.pid 2>/dev/null || true
bundle exec rails db:prepare
exec "$@"
