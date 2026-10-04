"""``python -m src.figures EXP_DIR [--only NAME ...] [--times T ...] [--out DIR] [--fmt pdf]``"""

import argparse
import logging

from . import FIGURES, load_run, make_figures


def main(argv=None):
    p = argparse.ArgumentParser(description="Render manuscript figures for one experiment directory.")
    p.add_argument("exp_dir")
    p.add_argument("--only", nargs="+", choices=sorted(FIGURES))
    p.add_argument("--times", nargs="+", type=float, metavar="T")
    p.add_argument("--out")
    p.add_argument("--fmt", default="pdf")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run = load_run(args.exp_dir)
    make_figures(run, names=args.only, out_dir=args.out, fmt=args.fmt,
                 times=tuple(args.times) if args.times else None)


if __name__ == "__main__":
    main()
