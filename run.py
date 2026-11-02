#!/usr/bin/env python3
"""Single entry point:  python3 run.py   (then open http://127.0.0.1:8000/)

Starts the web server. On first start it generates the synthetic belt video (+ ground truth),
runs the tracker and the evaluator, showing progress on the page; later starts reuse data/default.
"""
import sys

from conveyor.server import main

if __name__ == "__main__":
    main()
