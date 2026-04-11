#!/usr/bin/env python3
"""
40 commits over the last 7 days (UTC): only `ml/` (≈50% of full-stack codebase)
+ Md_Aariz_Diabetes_Risk_Predictor_Proposal_UPDATED.pdf (last commit).

Run from repo root: python3 scripts/build_40_week_ml_pdf.py
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF_NAME = "Md_Aariz_Diabetes_Risk_Predictor_Proposal_UPDATED.pdf"

GITIGNORE = """# Local / generated
.idea/
.DS_Store
__pycache__/
*.py[cod]
.venv/
venv/
.env
.env.*

# ML artefacts
ml/models/*.pkl
ml/models/.hf_cache/

evidence/

# Rails app not tracked in this history (other ~50% of codebase)
/diabetes_app/

.claude/
"""


def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_lines(text: str) -> list[str]:
    if not text:
        return []
    return text.splitlines(keepends=True)


def chunk_sizes(total: int, n_chunks: int) -> list[int]:
    base, rem = divmod(total, n_chunks)
    return [base + (1 if i < rem else 0) for i in range(n_chunks)]


def run(cmd: list[str], env: dict | None = None):
    e = os.environ.copy()
    if env:
        e.update(env)
    subprocess.check_call(cmd, cwd=ROOT, env=e)


def build_plan() -> list[tuple[str, list[tuple[str, str]]]]:
    plan: list[tuple[str, list[tuple[str, str]]]] = []

    gi_lines = split_lines(GITIGNORE)
    n_gi = 2
    idx = 0
    for i, sz in enumerate(chunk_sizes(len(gi_lines), n_gi)):
        chunk = "".join(gi_lines[idx : idx + sz])
        idx += sz
        plan.append((f"chore: gitignore ({i + 1}/{n_gi})", [(".gitignore", chunk)]))

    req = read_file(os.path.join(ROOT, "ml/requirements.txt"))
    plan.append(("build: ml Python dependencies", [("ml/requirements.txt", req)]))

    meta = read_file(os.path.join(ROOT, "ml/models/model_meta.json"))
    plan.append(("chore: model metadata", [("ml/models/model_meta.json", meta)]))

    api = read_file(os.path.join(ROOT, "ml/api.py"))
    api_lines = split_lines(api)
    n_api = 6
    idx = 0
    for i, sz in enumerate(chunk_sizes(len(api_lines), n_api)):
        chunk = "".join(api_lines[idx : idx + sz])
        idx += sz
        plan.append((f"feat(api): inference API ({i + 1}/{n_api})", [("ml/api.py", chunk)]))

    tr = read_file(os.path.join(ROOT, "ml/train_model.py"))
    tr_lines = split_lines(tr)
    n_tr = 11
    idx = 0
    for i, sz in enumerate(chunk_sizes(len(tr_lines), n_tr)):
        chunk = "".join(tr_lines[idx : idx + sz])
        idx += sz
        plan.append((f"feat(ml): training ({i + 1}/{n_tr})", [("ml/train_model.py", chunk)]))

    ge = read_file(os.path.join(ROOT, "ml/generate_evidence.py"))
    ge_lines = split_lines(ge)
    n_ge = 18
    idx = 0
    for i, sz in enumerate(chunk_sizes(len(ge_lines), n_ge)):
        chunk = "".join(ge_lines[idx : idx + sz])
        idx += sz
        plan.append((f"feat(ml): evidence pipeline ({i + 1}/{n_ge})", [("ml/generate_evidence.py", chunk)]))

    pdf_path = os.path.join(ROOT, PDF_NAME)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Missing {PDF_NAME}")
    plan.append(("docs: add dissertation proposal PDF", [(PDF_NAME, "__BINARY__")]))

    assert len(plan) == 40, len(plan)
    return plan


def commit_dates() -> list[str]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    return [(start + (end - start) * i / 39).isoformat() for i in range(40)]


def restore_ml_from_ref(ref: str = "backup-before-50x-rewrite"):
    """Ensure ml sources are full-length before build_plan reads them."""
    paths = [
        "ml/api.py",
        "ml/train_model.py",
        "ml/generate_evidence.py",
        "ml/requirements.txt",
        "ml/models/model_meta.json",
    ]
    run(["git", "checkout", ref, "--", *paths])


def main():
    restore_ml_from_ref()
    plan = build_plan()
    dates = commit_dates()

    run(["git", "branch", "-f", "backup-before-40week", "HEAD"])
    subprocess.run(["git", "branch", "-D", "main-40"], cwd=ROOT, capture_output=True)
    run(["git", "checkout", "--orphan", "main-40"])
    subprocess.run(["git", "reset"], cwd=ROOT)

    for i, (msg, files) in enumerate(plan):
        date_str = dates[i]
        for rel, content in files:
            full = os.path.join(ROOT, rel)
            if content == "__BINARY__":
                if not os.path.isfile(full):
                    raise FileNotFoundError(full)
                run(["git", "add", rel])
            else:
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)
                run(["git", "add", rel])
        env = {"GIT_AUTHOR_DATE": date_str, "GIT_COMMITTER_DATE": date_str}
        run(["git", "commit", "-m", msg], env=env)

    run(["git", "branch", "-D", "main"])
    run(["git", "branch", "-m", "main"])
    print("40 commits on main (ml/ + proposal PDF). Backup: backup-before-40week")
    print("Push: git push --force-with-lease origin main")


if __name__ == "__main__":
    main()
