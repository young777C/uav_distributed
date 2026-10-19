"""acot — single entry point for ACoT-UAV-Track data / prep / training / closed-loop evaluation / analysis
(design: acot_note/1004_code_framework.md).  `python -m acot --help`"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # repository root
CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
