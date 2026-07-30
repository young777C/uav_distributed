"""Layer-16 grounding probe for VLM backbone selection (design doc §3.4).

Compares frozen VLM backbones (PaliGemma-2-3B vs Qwen3-VL-4B) by how well their
per-layer representations let a tiny head SELECT the language-referred target
vehicle among visually-similar distractors — the capability the ACoT-UAV-Track
architecture must read out of a frozen, layer-16-truncated backbone.
"""
