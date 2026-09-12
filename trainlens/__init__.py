"""Offline, dependency-free comparison reports for training CSV logs."""

from .report import load_run, render_report, summarize

__all__ = ["load_run", "render_report", "summarize"]
