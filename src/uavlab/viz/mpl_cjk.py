from __future__ import annotations

import logging
import platform
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

logger = logging.getLogger(__name__)

_FONT_FAMILIES = (
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
)


def _existing_family_names() -> set[str]:
    return {f.name for f in font_manager.fontManager.ttflist}


def _candidate_font_files() -> list[Path]:
    paths: list[Path] = []
    system = platform.system()

    for p in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ):
        paths.append(Path(p))

    if system == "Windows":
        win = Path(r"C:\Windows\Fonts")
        paths.extend([win / "msyh.ttc", win / "simhei.ttf", win / "simsun.ttc"])
    else:
        wsl_win = Path("/mnt/c/Windows/Fonts")
        if wsl_win.is_dir():
            paths.extend([wsl_win / "msyh.ttc", wsl_win / "simhei.ttf", wsl_win / "simsun.ttc"])

    if system == "Darwin":
        paths.extend(
            [
                Path("/System/Library/Fonts/PingFang.ttc"),
                Path("/Library/Fonts/Arial Unicode.ttf"),
            ]
        )

    return paths


def configure_cjk_matplotlib(*, warn_if_missing: bool = True) -> str | None:
    """Register a CJK-capable sans font and update matplotlib rcParams."""
    available = _existing_family_names()
    for family in _FONT_FAMILIES:
        if family in available:
            plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return family

    for path in _candidate_font_files():
        if not path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
        except OSError:
            continue
        prop = font_manager.FontProperties(fname=str(path))
        family = prop.get_name()
        plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return family

    if warn_if_missing:
        logger.warning(
            "No CJK font found; Chinese text may render as squares. "
            "On Linux install fonts-noto-cjk; on WSL ensure /mnt/c/Windows/Fonts is available."
        )
    return None
