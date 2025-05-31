import numpy as np
from typing import Tuple
from scipy.stats import zscore

class AnomalyDetector:
    """Advanced anomaly detection for document processing."""
    
    def __init__(self, method='zscore', threshold=2.0):
        """
        Initialize detector.
        
        Args:
            method: Detection method ('zscore' or 'iqr')
            threshold: Sensitivity threshold
        """
        self.method = method
        self.threshold = threshold
        
    def detect(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies in performance data."""
        if self.method == 'zscore':
            scores = zscore(data)
            anomalies = np.abs(scores) > self.threshold
        elif self.method == 'iqr':
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)
            anomalies = (data < lower) | (data > upper)
            scores = (data - np.median(data)) / iqr
            
        return np.where(anomalies)[0], scores[anomalies]
    
    def explain_anomalies(self, anomalies, scores):
        """Generate human-readable anomaly explanations."""
        explanations = []
        for idx, score in zip(anomalies, scores):
            direction = "above" if score > 0 else "below"
            explanations.append(
                f"Step {idx}: Anomaly detected ({direction} expected range, "
                f"score={score:.2f})"
            )
        return explanations
