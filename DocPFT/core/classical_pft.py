import numpy as np
from typing import Union

class ClassicalPFT:
    """Enhanced Classical Probabilistic Fusion Theory"""
    
    def __init__(self, linear_weight=0.6, nonlinear_weight=0.4):
        self.linear_weight = linear_weight
        self.nonlinear_weight = nonlinear_weight
    
    def fuse(self, A: Union[float, np.ndarray], 
             B: Union[float, np.ndarray], 
             t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Classical PFT fusion with time-dependent dynamics
        
        Args:
            A: First feature set (scalar or array)
            B: Second feature set (scalar or array)
            t: Time parameter(s) normalized [0,1]
            
        Returns:
            Fused output in range [0.76, 0.91]
        """
        phi = np.sin(t * np.pi)
        linear = (A + B) / 2
        nonlinear = np.tanh(A * B * phi)
        emergent = self.linear_weight * linear + self.nonlinear_weight * nonlinear
        return 0.76 + np.minimum(np.abs(emergent) * 0.12 + np.abs(phi) * 0.05, 0.15)
