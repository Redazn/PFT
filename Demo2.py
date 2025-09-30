#SAMPLING THEORY applied to Rubrik's Cube

class ShannonRubrikOptimizer:
def init(self, large_matrix, sampling_method='fourier', target_size=(3,3)):
"""
large_matrix: np.array (bisa 10x10, 20x20, dll)
sampling_method: 'fourier', 'wavelet', 'max_pool', 'information'
target_size: (3,3) untuk maintain Rubrik's cube philosophy
"""
self.original_matrix = large_matrix
self.sampling_method = sampling_method
self.target_size = target_size

def compress_to_rubrik_format(self):  
    """Kompresi data besar ke 3x3 dengan retain informasi maksimal"""  
      
    if self.sampling_method == 'fourier':  
        return self._fourier_sampling()  
          
    elif self.sampling_method == 'wavelet':  
        return self._wavelet_compression()  
          
    elif self.sampling_method == 'max_pool':  
        return self._max_pool_sampling()  
          
    elif self.sampling_method == 'information':  
        return self._information_theory_sampling()  
          
    elif self.sampling_method == 'musical':  
        return self._vinyl_to_digital_sampling()  

def _fourier_sampling(self):  
    """Aliasing frequency domain - ambil low-frequency components"""  
    # FFT → ambil coefficients terpenting → inverse FFT ke 3x3  
    fft_data = np.fft.fft2(self.original_matrix)  
    # Keep low frequencies (center of FFT)  
    rows, cols = self.original_matrix.shape  
    center_r, center_c = rows//2, cols//2  
    # Extract 3x3 low-frequency components  
    compressed_fft = fft_data[center_r-1:center_r+2, center_c-1:center_c+2]  
    return np.abs(np.fft.ifft2(compressed_fft))  

def _wavelet_compression(self):  
    """Wavelet transform - keep approximation coefficients"""  
    import pywt  
    # Wavelet decomposition  
    coeffs = pywt.wavedec2(self.original_matrix, 'haar', level=3)  
    # Keep only approximation coefficients at appropriate level  
    cA = coeffs[0]  # Approximation coefficients  
    # Resize to 3x3  
    from scipy.ndimage import zoom  
    return zoom(cA, (3/cA.shape[0], 3/cA.shape[1]))  

def _information_theory_sampling(self):  
    """Shannon information theory - maximize retained information"""  
    # Hitung information content per region  
    matrix = self.original_matrix  
    row_splits = np.array_split(matrix, 3, axis=0)  
    compressed = []  
      
    for row_chunk in row_splits:  
        col_splits = np.array_split(row_chunk, 3, axis=1)  
        row_compressed = []  
        for col_chunk in col_splits:  
            # Pilih nilai yang maximize information (bukan average)  
            # Gunakan entropy sebagai measure of information  
            entropy_vals = [self._cell_entropy(col_chunk, i, j)   
                          for i in range(col_chunk.shape[0])   
                          for j in range(col_chunk.shape[1])]  
            # Ambil cell dengan highest entropy (most information)  
            max_idx = np.argmax(entropy_vals)  
            i, j = max_idx // col_chunk.shape[1], max_idx % col_chunk.shape[1]  
            row_compressed.append(col_chunk[i, j])  
        compressed.append(row_compressed)  
          
    return np.array(compressed)  

def _vinyl_to_digital_sampling(self):  
    """Analog to digital conversion style sampling"""  
    # Sample seperti ADC (Analog-to-Digital Converter)  
    matrix = self.original_matrix  
      
    # 1. Quantization: reduce amplitude resolution  
    quantized = np.digitize(matrix, bins=np.linspace(matrix.min(), matrix.max(), 8))  
      
    # 2. Sampling: take strategic samples (bukan uniform)  
    # Sample points based on gradient importance  
    grad_y, grad_x = np.gradient(matrix)  
    importance_map = np.abs(grad_y) + np.abs(grad_x)  
      
    # Pilih 9 points dengan highest importance (bukan uniform grid)  
    flat_importance = importance_map.flatten()  
    top_indices = np.argsort(flat_importance)[-9:][::-1]  
      
    sampled_values = matrix.flatten()[top_indices]  
    return sampled_values.reshape(3, 3)  

def _cell_entropy(self, chunk, i, j):  
    """Calculate local entropy around cell (i,j)"""  
    from scipy.stats import entropy  
    # Take 3x3 neighborhood around cell  
    i_start, i_end = max(0, i-1), min(chunk.shape[0], i+2)  
    j_start, j_end = max(0, j-1), min(chunk.shape[1], j+2)  
    neighborhood = chunk[i_start:i_end, j_start:j_end]  
      
    if neighborhood.size == 0:  
        return 0  
          
    flat = neighborhood.flatten()  
    hist, _ = np.histogram(flat, bins=min(8, len(flat)))  
    prob = hist / hist.sum()  
    prob = prob[prob > 0]  # Remove zeros  
    return entropy(prob)

paham ga kenapa saya kirim kode ini

