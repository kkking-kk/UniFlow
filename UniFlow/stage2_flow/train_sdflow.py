"""Backward-compatible Stage-2 training entry point.

Prefer: python stage2_flow/train_uniflow.py [arguments]
"""
import os
import runpy

if __name__ == "__main__":
    runpy.run_path(
        os.path.join(os.path.dirname(__file__), "train_uniflow.py"),
        run_name="__main__",
    )
