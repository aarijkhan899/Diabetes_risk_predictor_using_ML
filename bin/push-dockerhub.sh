#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROJECT_SLUG="aariz"
ML_IMAGE="${DOCKERHUB_USER}/${PROJECT_SLUG}-ml-api:latest"
RAILS_IMAGE="${DOCKERHUB_USER}/${PROJECT_SLUG}-rails-app:latest"

detect_dockerhub_user() {
  local config="${HOME}/.docker/config.json"
  [[ -f "$config" ]] || return 1
  command -v python3 >/dev/null 2>&1 || return 1

  python3 - "$config" <<'PY'
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

path = Path(sys.argv[1])
try:
    cfg = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)

def user_from_auth_blob(blob: str) -> Optional[str]:
    if not blob:
        return None
    try:
        raw = base64.b64decode(blob).decode("utf-8", errors="replace")
    except Exception:
        return None
    user = raw.split(":", 1)[0].strip()
    return user or None

helpers = cfg.get("credHelpers") or {}
helper = helpers.get("docker.io") or helpers.get("https://index.docker.io/v1/")
if helper:
    exe = shutil.which(f"docker-credential-{helper}")
    if exe:
        for url in (
            "https://index.docker.io/v1/",
            "https://index.docker.io/v1",
            "https://docker.io",
        ):
            try:
                proc = subprocess.run(
                    [exe, "get"],
                    input=json.dumps({"ServerURL": url}),
                    text=True,
                    capture_output=True,
                    timeout=20,
                    check=False,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    data = json.loads(proc.stdout)
                    u = (data.get("Username") or "").strip()
                    if u:
                        print(u)
                        raise SystemExit(0)
            except Exception:
                pass

store = (cfg.get("credsStore") or "").strip()
if store:
    exe = shutil.which(f"docker-credential-{store}")
    if exe:
        for url in (
            "https://index.docker.io/v1/",
            "https://index.docker.io/v1",
            "https://docker.io",
        ):
            try:
                proc = subprocess.run(
                    [exe, "get"],
                    input=json.dumps({"ServerURL": url}),
                    text=True,
                    capture_output=True,
                    timeout=20,
                    check=False,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    data = json.loads(proc.stdout)
                    u = (data.get("Username") or "").strip()
                    if u:
                        print(u)
                        raise SystemExit(0)
            except Exception:
                pass

auths = cfg.get("auths") or {}
for key in (
    "https://index.docker.io/v1/",
    "https://index.docker.io/v1",
    "https://docker.io",
    "https://registry-1.docker.io",
):
    u = user_from_auth_blob((auths.get(key) or {}).get("auth"))
    if u:
        print(u)
        raise SystemExit(0)

sys.exit(1)
PY
}

if [[ -z "${DOCKERHUB_USER:-}" ]]; then
  if detected="$(detect_dockerhub_user)"; then
    export DOCKERHUB_USER="$detected"
    echo "Using DOCKERHUB_USER from Docker config: $DOCKERHUB_USER"
  else
    echo "Set DOCKERHUB_USER (e.g. export DOCKERHUB_USER=isammalik) or run \`docker login\`." >&2
    exit 1
  fi
fi

ML_IMAGE="${DOCKERHUB_USER}/${PROJECT_SLUG}-ml-api:latest"
RAILS_IMAGE="${DOCKERHUB_USER}/${PROJECT_SLUG}-rails-app:latest"

BUILDER_NAME="${BUILDX_BUILDER:-aariz-multiarch}"

if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
  docker buildx create --name "$BUILDER_NAME" --driver docker-container --use
else
  docker buildx use "$BUILDER_NAME"
fi

echo "Building and pushing: $ML_IMAGE"
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --provenance=false \
  --push \
  -f Dockerfile.ml \
  -t "$ML_IMAGE" \
  .

echo "Building and pushing: $RAILS_IMAGE"
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --provenance=false \
  --push \
  -f Dockerfile.rails \
  -t "$RAILS_IMAGE" \
  .

echo ""
echo "Set each repository to Public on Docker Hub so anonymous pulls work:"
echo "  https://hub.docker.com/repository/docker/${DOCKERHUB_USER}/${PROJECT_SLUG}-ml-api/settings"
echo "  https://hub.docker.com/repository/docker/${DOCKERHUB_USER}/${PROJECT_SLUG}-rails-app/settings"
echo ""
echo "Pull and run locally:"
echo "  export DOCKERHUB_USER=${DOCKERHUB_USER}"
echo "  docker compose -f docker-compose.hub.yml pull && docker compose -f docker-compose.hub.yml up"
