"""distill_lib — modular distillation library.

Public API re-exports for convenient access from external code.
"""

from distill_lib.encoder import ImageEncoderWrapper
from distill_lib.student import build_student_config, load_student, load_student_from_checkpoint
from distill_lib.teacher import load_teacher
