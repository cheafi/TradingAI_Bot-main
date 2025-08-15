# tests/test_risk.py
import numpy as np
from src.utils.risk import kelly_fraction, sharpe

def test_kelly_basic():
    f = kelly_fraction(0.6, 1.5, cap=0.5)
    assert 0 <= f <= 0.5

def test_sharpe_zero():
    r = np.zeros(10)
    assert sharpe(r) == 0.0
