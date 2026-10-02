"""Figure registry: decorate a function of a ``Run`` and it becomes available by name."""

import inspect
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from .data import Run
from .poster_style import save_figure, style_context

logger = logging.getLogger(__name__)

FIGURES: dict = {}


def figure(name: str):
    def wrap(fn):
        FIGURES[name] = fn
        return fn
    return wrap


def make_figures(run: Run, names=None, out_dir=None, fmt: str = "pdf", **options) -> dict:
    """Render the named figures (default: all). ``options`` go to every figure that accepts them."""
    out_dir = Path(out_dir) if out_dir else run.exp_dir / "figures"
    unknown = set(names or []) - set(FIGURES)
    if unknown:
        raise KeyError(f"Unknown figures {sorted(unknown)}; available: {sorted(FIGURES)}")

    written = {}
    for name in names or FIGURES:
        fn = FIGURES[name]
        accepted = inspect.signature(fn).parameters
        kwargs = {k: v for k, v in options.items() if k in accepted and v is not None}
        with style_context():
            fig = fn(run, **kwargs)
            written[name] = save_figure(fig, out_dir / f"{name}.{fmt}")
        plt.close(fig)
        logger.info("Wrote %s", written[name])
    return written
