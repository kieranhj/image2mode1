#!/usr/bin/env python3
"""Entry point for Hugging Face Spaces (or local launch from repo root)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "palsearch"))

from gradio_ui import build_ui

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
