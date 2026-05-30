"""Distill UNI vision encoder (ViT-H/14, 1536-dim) into a smaller student ViT.

Standalone training script with periodic linear-probe evaluation on CRC-100K.
See distill.md for the full specification.
"""

from distill_lib.main import main

if __name__ == "__main__":
    main()
