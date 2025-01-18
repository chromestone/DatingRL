"""
types.py

This module stores types to be used by datingrl.
"""

from typing import Optional

import numpy as np

ACTIONS_PROBS_TUPLE = tuple[np.ndarray[np.float32], Optional[np.ndarray[np.float32]]]
INT_FLOAT_TUPLE = tuple[int, Optional[float]]
