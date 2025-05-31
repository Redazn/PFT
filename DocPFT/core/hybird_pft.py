from typing import Tuple, Optional
import numpy as np
from .classical_pft import ClassicalPFT
from .q_inspired_pft import QuantumPFT

class HybridPFT:
    def __init__(self, time_window=20):
        self.classical = ClassicalPFT()
        self.quantum = QuantumPFT()
        self.time_window = time_window
        self.memory = []
        
    def adaptive_weights(self, entropy):
        quantum_weight = min(0.7, 0.3 + 0.5 * entropy)
        return 1 - quantum_weight, quantum_weight
        
    def process(self, feature, t, q_params):
        # Update memory for temporal context
        self.memory.append(feature)
        if len(self.memory) > self.time_window:
            self.memory.pop(0)
            
        # Classical processing (current feature)
        classical_out = self.classical.process(feature, t)
        
        # Quantum processing (contextual features)
        context_feature = np.mean(self.memory, axis=0) if self.memory else feature
        quantum_out = self.quantum.process(context_feature, t, q_params)
        
        # Adaptive weighting
        entropy_val = self.classical.compute_entropy()
        alpha, beta = self.adaptive_weights(entropy_val)
        hybrid = alpha * classical_out + beta * quantum_out
        return classical_out, quantum_out, hybrid
