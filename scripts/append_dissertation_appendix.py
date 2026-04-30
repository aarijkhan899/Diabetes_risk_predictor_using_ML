#!/usr/bin/env python3
"""
Append a professionally formatted appendix (code listings + verified sample outputs)
to the combined dissertation DOCX.

Usage:
  python scripts/append_dissertation_appendix.py "Aariz_Final_Combined_Dissertation (1).docx"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from docx import Document
    from docx.enum.text import WD_BREAK
    from docx.shared import Inches, Pt
except ImportError as e:
    print("Requires python-docx: pip install python-docx", file=sys.stderr)
    raise SystemExit(1) from e

ROOT = Path(__file__).resolve().parents[1]


def monospace_block(doc: Document, body: str) -> None:
    para = doc.add_paragraph()
    fmt = para.paragraph_format
    fmt.left_indent = Inches(0.28)
    fmt.space_before = Pt(4)
    fmt.space_after = Pt(10)
    run = para.add_run(body.strip("\n"))
    run.font.name = "Courier New"
    run.font.size = Pt(9)


def prose(doc: Document, text: str) -> None:
    para = doc.add_paragraph(text)
    para.paragraph_format.space_after = Pt(8)


def bold_title(doc: Document, text: str, *, size_pt: int = 14) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(size_pt)
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(8)


def collect_sample_json(repo_root: Path) -> tuple[str, str, str]:
    """Run Flask test_client; return meta file, GET /health, POST /predict JSON."""
    cwd = repo_root

    snippet = '''
import logging, json, importlib.util
from pathlib import Path

logging.disable(logging.CRITICAL)
spec = importlib.util.spec_from_file_location("api", Path("ml") / "api.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = mod.app
c = app.test_client()

meta_txt = Path("ml/models/model_meta.json").read_text(encoding="utf-8")
health = json.dumps(c.get("/health").get_json(), indent=2)

body = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}
pred_j = json.dumps(c.post("/predict", json=body).get_json(), indent=2)
print("__META_START__")
print(meta_txt.strip())
print("__META_END__")
print("__HEALTH_START__")
print(health.strip())
print("__HEALTH_END__")
print("__PREDICT_START__")
print(pred_j.strip())
print("__PREDICT_END__")
'''
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    out = proc.stdout
    err = proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"Sample run failed ({proc.returncode}): {err or out}")

    def block(tag: str) -> str:
        a = out.index(f"__{tag}_START__") + len(f"__{tag}_START__")
        b = out.index(f"__{tag}_END__")
        return out[a:b].strip()

    return block("META"), block("HEALTH"), block("PREDICT")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", nargs="?", default=str(ROOT / "Aariz_Final_Combined_Dissertation (1).docx"))
    args = ap.parse_args()
    path = Path(args.docx).resolve()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    doc = Document(str(path))

    if any(
        "appendix — implementation listing" in (p.text or "").lower()
        or "implementation listing & sample artefacts" in (p.text or "").lower()
        for p in doc.paragraphs
    ):
        raise SystemExit(
            "Appendix already present in this document (refusing to duplicate). "
            "Restore from the *_backup_before_appendix.docx copy if needed, then edit and re-run."
        )

    meta_js, health_js, predict_js = collect_sample_json(ROOT)

    tl = (ROOT / "ml/train_model.py").read_text(encoding="utf-8").splitlines()
    train_slice = tl[99:173]
    train_code = "\n".join(train_slice).rstrip() + "\n\n# ... Evaluation, artefact persistence, main() trimmed ..."

    al = (ROOT / "ml/api.py").read_text(encoding="utf-8").splitlines()
    api_code = "\n".join(al[20:200]).strip()
    api_code += "\n\n# ... Flask application entry trimmed ..."

    rl = (
        ROOT / "diabetes_app/app/controllers/predictions_controller.rb"
    ).read_text(encoding="utf-8").splitlines()

    bold_title(doc, "Appendix — Implementation listing & sample artefacts", size_pt=16)

    prose(
        doc,
        "This appendix aligns with the implemented dissertation prototype (Python ML stack + Flask REST API + Ruby on Rails frontend). "
        "Listings are taken from the project repository verbatim and typeset monospaced. JSON samples below were emitted by invoking the Flask "
        "application’s bundled test_client after loading persisted joblib models (representative artefacts; re-training refreshes numerical fields). ",
    )

    bold_title(doc, "Appendix A — Preprocessing & model selection excerpt", size_pt=12)

    prose(
        doc,
        "Representative excerpt from ml/train_model.py: zero-value imputation for clinically invalid sentinel values on selected fields, StandardScaler "
        "fit-transform, SMOTE class balancing on the stratified folds, followed by ROC-AUC–scored GridSearchCV across heterogeneous estimators.",
    )
    monospace_block(doc, train_code)

    bold_title(doc, "Appendix B — REST API: artefacts, health check, inference", size_pt=12)

    prose(
        doc,
        "Excerpt from ml/api.py: contractual feature ordering, artefact hydration, probabilistic inference, SHAP-based local explanations, JSON contract including disclaimer fields.",
    )
    monospace_block(doc, api_code)

    bold_title(doc, "Appendix C — Rails orchestration HTTP client", size_pt=12)

    prose(doc, "Excerpt from app/controllers/predictions_controller.rb: server-side validation, HTTParty JSON POST to ML_API_URL, error handling.")

    monospace_block(doc, rails_code)

    bold_title(doc, "Appendix D — Persisted classifier metadata snapshot", size_pt=12)

    prose(doc, "Current ml/models/model_meta.json.")
    monospace_block(doc, meta_js)

    bold_title(doc, "Appendix E — Sample API payloads (automated client)", size_pt=12)

    prose(
        doc,
        "Captured JSON for GET /health and POST /predict using the canonical Pima demonstration instance identical to Rails EXAMPLE_PATIENT.",
    )
    monospace_block(
        doc,
        "GET /health\n==\n" + health_js + "\n\nPOST /predict\n==\n" + predict_js,
    )

    doc.save(str(path))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
