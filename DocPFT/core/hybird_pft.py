from typing import Tuple, Optional
import numpy as np
from .classical_pft import ClassicalPFT
from .q_inspired_pft import QuantumPFT

class HybridPFT:
    """Hybrid Classical-Quantum Fusion System"""
    
    def __init__(self, classical_weight=0.7, quantum_weight=0.3):
        self.classical = ClassicalPFT()
        self.quantum = QuantumPFT()
        self.weights = {
            'classical': classical_weight,
            'quantum': quantum_weight
        }
        
        # Normalize weights
        total = classical_weight + quantum_weight
        self.weights['classical'] /= total
        self.weights['quantum'] /= total
    
    def fuse(self, A: np.ndarray, B: np.ndarray, 
            n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Hybrid fusion over multiple time steps
        
        Returns:
            Tuple of (classical_results, quantum_results, hybrid_results)
        """
        classical_results = []
        quantum_results = []
        
        for i in range(n_steps):
            t = i / n_steps
            classical = self.classical.fuse(A[i], B[i], t)
            quantum = self.quantum.fuse(A[i], B[i], t)
            
            classical_results.append(classical)
            quantum_results.append(quantum)
        
        hybrid = (self.weights['classical'] * np.array(classical_results) + 
                 self.weights['quantum'] * np.array(quantum_results))
        
        return classical_results, quantum_results, hybrid
