"""Non-interactive driver for heretic-llm v1.2.0.

heretic v1.2 is built around questionary (interactive CLI). Our pipeline
needs full automation. This wrapper monkey-patches heretic's interactive
prompts to return preset answers, then calls heretic.main.main() with the
desired settings via sys.argv.

Auto-responses:
- "How would you like to proceed?" (existing study) → "restart"
- "Which trial do you want to use?"                 → first (lowest refusals/KL)
- "What do you want to do with the decensored model?" → "Save the model to a local folder"
- "Path to the folder:"                              → SAVE_DIR (env)
- merge strategy                                     → "merge"

Usage:
    SAVE_DIR=/data/out/aggressive python run_heretic_auto.py \\
      --model /data/out/merged --n-trials 100
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Inject these BEFORE importing heretic so that monkey-patches stick.
SAVE_DIR = os.environ.get("SAVE_DIR", "/data/out/aggressive")
os.makedirs(SAVE_DIR, exist_ok=True)


def patch_heretic() -> None:
    """Replace interactive prompts with preset values."""
    import heretic.utils as hu

    original_prompt_select = hu.prompt_select
    original_prompt_path = hu.prompt_path
    original_prompt_text = hu.prompt_text

    def auto_select(message: str, choices, **kwargs):  # type: ignore[no-untyped-def]
        msg = message.lower() if isinstance(message, str) else ""
        # Surface what we're auto-answering, for log clarity.
        labels = []
        for c in choices:
            if hasattr(c, "title"):
                labels.append(str(c.title))
            else:
                labels.append(str(c))
        print(f"[auto] prompt_select: {message!r}")
        print(f"[auto]   choices: {labels}")

        # 1. Existing-study prompt
        if "how would you like to proceed" in msg:
            for c in choices:
                if getattr(c, "value", None) == "restart":
                    print("[auto]   -> restart")
                    return c.value
            return choices[0].value if hasattr(choices[0], "value") else choices[0]

        # 2. Pareto trial selection — pick the FIRST trial (lowest refusals, lowest KL among same)
        if "which trial" in msg:
            for c in choices:
                v = getattr(c, "value", None)
                if hasattr(v, "user_attrs"):
                    print(f"[auto]   -> trial {v.user_attrs.get('index','?')} "
                          f"refusals={v.user_attrs.get('refusals')} "
                          f"kl={v.user_attrs.get('kl_divergence'):.4f}")
                    return v
            print(f"[auto]   -> {labels[0]} (first choice)")
            return choices[0].value if hasattr(choices[0], "value") else choices[0]

        # 3. Action menu — save to local folder
        if "what do you want to do" in msg:
            for c in choices:
                title = c if isinstance(c, str) else getattr(c, "title", str(c))
                if "save" in str(title).lower():
                    print(f"[auto]   -> {title}")
                    return c if isinstance(c, str) else c.value
            return choices[0]

        # 4. Merge strategy (only triggers if BNB_4BIT) — choose merge
        if "how do you want to proceed" in msg:
            for c in choices:
                v = getattr(c, "value", None) if hasattr(c, "value") else c
                if v == "merge":
                    print("[auto]   -> merge")
                    return v
            return choices[0]

        # 5. Visibility — public
        if "public or private" in msg:
            print("[auto]   -> Public")
            return "Public"

        print(f"[auto]   FALLBACK -> {labels[0]}")
        return choices[0].value if hasattr(choices[0], "value") else choices[0]

    def auto_path(message: str, **kwargs):  # type: ignore[no-untyped-def]
        print(f"[auto] prompt_path: {message!r} -> {SAVE_DIR}")
        return SAVE_DIR

    def auto_text(message: str, **kwargs):  # type: ignore[no-untyped-def]
        # Used for "How many additional trials" — we never want to add more
        if "additional trials" in (message or "").lower():
            print(f"[auto] prompt_text: {message!r} -> 0 (don't add trials)")
            return "0"
        # Used for repo name — we save locally, so this shouldn't fire
        default = kwargs.get("default", "")
        print(f"[auto] prompt_text: {message!r} -> {default}")
        return default

    hu.prompt_select = auto_select
    hu.prompt_path = auto_path
    hu.prompt_text = auto_text

    # heretic/main.py imports these names directly: shadow them in main module too
    import heretic.main as hm
    hm.prompt_select = auto_select
    hm.prompt_path = auto_path
    hm.prompt_text = auto_text


def main() -> int:
    patch_heretic()
    # Forward all CLI args to heretic.main; user controls via env / argv
    from heretic.main import main as heretic_main
    return heretic_main() or 0


if __name__ == "__main__":
    sys.exit(main())
