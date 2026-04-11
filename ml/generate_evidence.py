#!/usr/bin/env python3
"""Placeholder: training metrics and figures are produced by `train_model.py` (see `evidence/` if enabled)."""

import sys


def main() -> None:
    print(
        "Evidence export is integrated with training. Run:\n"
        "  cd ml && python train_model.py\n"
        "Model outputs go to ml/models/; add plotting in train_model.py if you need charts.",
        file=sys.stderr,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
