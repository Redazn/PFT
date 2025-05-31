import numpy as np
from typing import Union

# ================================
# [2] PFT AGENTS (IMPROVED)
# ================================
class ClassicalPFT:
    def __init__(self, temperature=0.5):
        self.T = temperature
        self.entropy_window = []
        self.window_size = 10
        
    def update_entropy(self, value):
        self.entropy_window.append(value)
        if len(self.entropy_window) > self.window_size:
            self.entropy_window.pop(0)
            
    def compute_entropy(self):
        if len(self.entropy_window) < 2:
            return 0
        p = np.abs(self.entropy_window) / (np.sum(np.abs(self.entropy_window)) + 1e-10)
        return -np.sum(p * np.log(p + 1e-10))
    
    def process(self, feature, t):
        self.update_entropy(feature)
        entropy = self.compute_entropy()
        F = feature**2 - self.T * entropy
        nonlinear = np.tanh(F)
        if F < -0.2:
            nonlinear *= 1.8
        return 0.5 + 0.3 * nonlinear
