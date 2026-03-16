"""
Constant folding pass (prototype).

Goal:
  Precompute operations whose inputs are compile-time constants, replacing them
  with constant nodes to reduce runtime work and memory traffic.
"""

