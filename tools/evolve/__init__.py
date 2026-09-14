"""Profit-selected evolution of declared free parameters. Disclosed, not hidden.

Lives outside the stonkfly package on purpose: cli.py hashes every .py and
.cpp under stonkfly/ into provenance_sha256, so anything placed there would
invalidate every existing run directory. Nothing here is imported by the
runtime, and nothing here ever places an order.
"""
