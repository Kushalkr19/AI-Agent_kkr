#!/usr/bin/env python3
"""
Compatibility wrapper for the enhanced résumé agent.

This script preserves the original command‑line interface described in the
repository's documentation while delegating all logic to ``resume_agent.py``.
Running this file directly has the same effect as running ``resume_agent.py``.
"""
from resume_agent import main

if __name__ == '__main__':
    main()