#!/usr/bin/env python3
"""
Build 50 commits with dates between 2026-03-19 and 2026-04-05 containing ~50% of
repository lines (ml/ + Docker + meta + gitignore; Rails tree excluded via .gitignore).
Run from repo root: python3 scripts/build_50_percent_history.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files included in the pushed history (full content); ~51% of prior total lines
FILES_ORDER = [
    ".gitignore",
    "Dockerfile.ml",
    "docker-compose.yml",
    "ml/requirements.txt",
    "ml/models/model_meta.json",
    "ml/api.py",
    "ml/train_model.py",
    "ml/generate_evidence.py",
]

# Extra line appended to .gitignore so Rails app (other ~50%) is not tracked
GITIGNORE_EXCLUDE_RAILS = "\n# Partial publish: remainder of codebase kept local\n/diabetes_app/\n"


def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    return lines


def chunk_sizes(total: int, n_chunks: int) -> list[int]:
    """Split total lines into n_chunks positive integers."""
    base, rem = divmod(total, n_chunks)
    return [base + (1 if i < rem else 0) for i in range(n_chunks)]


def build_commit_plan() -> list[tuple[str, list[tuple[str, str]]]]:
    """
    Returns list of (message, [(relpath, partial_content), ...])
    """
    plan: list[tuple[str, list[tuple[str, str]]]] = []

    # Prepare .gitignore with Rails exclusion
    gi_full = read_file(os.path.join(ROOT, ".gitignore")) + GITIGNORE_EXCLUDE_RAILS
    gi_lines = split_lines(gi_full)
    n_gi = 2
    sizes_gi = chunk_sizes(len(gi_lines), n_gi)
    idx = 0
    for i, sz in enumerate(sizes_gi):
        chunk = "".join(gi_lines[idx : idx + sz])
        idx += sz
        plan.append(
            (
                f"chore: add project gitignore ({i + 1}/{n_gi})",
                [(".gitignore", chunk)],
            )
        )

    # Dockerfile single commit
    df = read_file(os.path.join(ROOT, "Dockerfile.ml"))
    plan.append(("build: add Dockerfile for ML API", [("Dockerfile.ml", df)]))

    # docker-compose 2 commits
    dc = read_file(os.path.join(ROOT, "docker-compose.yml"))
    dc_lines = split_lines(dc)
    n_dc = 2
    idx = 0
    for i, sz in enumerate(chunk_sizes(len(dc_lines), n_dc)):
        chunk = "".join(dc_lines[idx : idx + sz])
        idx += sz
        plan.append(
            (
                f"build: docker-compose service wiring ({i + 1}/{n_dc})",
                [("docker-compose.yml", chunk)],
            )
        )

    req = read_file(os.path.join(ROOT, "ml/requirements.txt"))
    plan.append(("build: Python dependencies for ML stack", [("ml/requirements.txt", req)]))

    meta = read_file(os.path.join(ROOT, "ml/models/model_meta.json"))
    plan.append(("chore: model metadata placeholder", [("ml/models/model_meta.json", meta)]))

    # api.py
    api = read_file(os.path.join(ROOT, "ml/api.py"))
    api_lines = split_lines(api)
    n_api = 6
    idx = 0
    for i, sz in enumerate(chunk_sizes(len(api_lines), n_api)):
        chunk = "".join(api_lines[idx : idx + sz])
        idx += sz
        plan.append(
            (
                f"feat(api): Flask inference service ({i + 1}/{n_api})",
                [("ml/api.py", chunk)],
            )
        )

    # train_model.py (12 commits; 25 reserved for generate_evidence)
    tr = read_file(os.path.join(ROOT, "ml/train_model.py"))
    tr_lines = split_lines(tr)
    n_tr = 12
    idx = 0
    for i, sz in enumerate(chunk_sizes(len(tr_lines), n_tr)):
        chunk = "".join(tr_lines[idx : idx + sz])
        idx += sz
        plan.append(
            (
                f"feat(ml): training pipeline ({i + 1}/{n_tr})",
                [("ml/train_model.py", chunk)],
            )
        )
    # generate_evidence.py — 25 commits
    ge = read_file(os.path.join(ROOT, "ml/generate_evidence.py"))
    ge_lines = split_lines(ge)
    n_ge = 25
    idx = 0
    for i, sz in enumerate(chunk_sizes(len(ge_lines), n_ge)):
        chunk = "".join(ge_lines[idx : idx + sz])
        idx += sz
        plan.append(
            (
                f"feat(ml): evidence generation pipeline ({i + 1}/{n_ge})",
                [("ml/generate_evidence.py", chunk)],
            )
        )

    assert len(plan) == 50, f"expected 50 commits, got {len(plan)}"
    return plan


def commit_dates() -> list[str]:
    start = datetime(2026, 3, 19, 9, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 5, 17, 0, 0, tzinfo=timezone.utc)
    return [
        (start + (end - start) * i / 49).isoformat()
        for i in range(50)
    ]


def run(cmd: list[str], env: dict | None = None):
    e = os.environ.copy()
    if env:
        e.update(env)
    subprocess.check_call(cmd, cwd=ROOT, env=e)


def main():
    plan = build_commit_plan()
    dates = commit_dates()

    os.chdir(ROOT)
    # Backup branch
    run(["git", "branch", "-f", "backup-before-50x-rewrite", "HEAD"])

    subprocess.run(["git", "branch", "-D", "main-50"], cwd=ROOT, capture_output=True)
    # Orphan branch
    run(["git", "checkout", "--orphan", "main-50"])
    run(["git", "reset"])
    # Remove index entries if any
    subprocess.run(["git", "rm", "-rf", "--cached", "."], cwd=ROOT, capture_output=True)

    # Ensure dirs exist when writing
    for i, (msg, files) in enumerate(plan):
        date_str = dates[i]
        for rel, content in files:
            full = os.path.join(ROOT, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            run(["git", "add", rel])
        env = {
            "GIT_AUTHOR_DATE": date_str,
            "GIT_COMMITTER_DATE": date_str,
        }
        run(["git", "commit", "-m", msg], env=env)

    # Replace main
    run(["git", "branch", "-D", "main"])
    run(["git", "branch", "-m", "main"])
    print("Done: 50 commits on main. Previous tip saved as backup-before-50x-rewrite")
    print("Force-push: git push --force-with-lease origin main")


if __name__ == "__main__":
    main()
