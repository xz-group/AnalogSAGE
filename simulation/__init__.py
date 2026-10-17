"""Testbench templates (one folder per circuit type) and the ngspice runner."""

from .ngspice import parse_check, parse_values, render_testbench, run_ngspice, write_varspice

__all__ = ["parse_check", "parse_values", "render_testbench", "run_ngspice", "write_varspice"]
