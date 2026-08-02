"""
Kalman Temporal Denoising Module for Satellite Thermal Time Series
Applies state-space temporal Kalman filtering across time steps (t_1, t_2, ..., t_N)
to reduce measurement noise, smooth thermal variations, and impute cloud-masked pixels.
"""

import numpy as np

class KalmanTemporalDenoiser:
    """
    1D/2D Discrete Kalman Filter for temporal thermal sequence denoising.
    
    State equation:   x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
    Measurement:      z_t = x_t + v_t,        v_t ~ N(0, R)
    """
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2, initial_covariance=1.0):
        self.q = process_variance       # Process noise covariance Q
        self.r = measurement_variance    # Measurement noise covariance R
        self.p_init = initial_covariance

    def filter_sequence(self, thermal_sequence: np.ndarray, cloud_masks: np.ndarray = None) -> np.ndarray:
        """
        Denoise a 3D/4D numpy array sequence (Time, Height, Width) or (Time, Channels, Height, Width).
        
        Args:
            thermal_sequence: np.ndarray of shape (T, H, W) or (T, C, H, W)
            cloud_masks: np.ndarray of boolean/uint8 masks (1=valid, 0=cloud/invalid) of shape matching (T, H, W)
            
        Returns:
            denoised_sequence: np.ndarray of same shape as thermal_sequence
        """
        seq = np.asarray(thermal_sequence, dtype=np.float32)
        is_4d = (seq.ndim == 4)
        if is_4d:
            T, C, H, W = seq.shape
        elif seq.ndim == 3:
            T, H, W = seq.shape
            C = 1
            seq = seq[:, np.newaxis, :, :]
        elif seq.ndim == 2:
            # Single spatial frame over time 1
            seq = seq[np.newaxis, np.newaxis, :, :]
            T, C, H, W = seq.shape
        else:
            raise ValueError(f"Unsupported sequence shape for Kalman filter: {seq.shape}")

        if cloud_masks is not None:
            masks = np.asarray(cloud_masks, dtype=bool)
            if masks.ndim == 3:
                masks = masks[:, np.newaxis, :, :]
        else:
            masks = np.ones((T, C, H, W), dtype=bool)

        denoised = np.zeros_like(seq)
        
        # State vector x and covariance P per pixel
        x_est = seq[0].copy()
        p_est = np.full((C, H, W), self.p_init, dtype=np.float32)

        for t in range(T):
            z_t = seq[t]
            valid_mask = masks[t]

            # Time Update (Prediction)
            x_pred = x_est
            p_pred = p_est + self.q

            # Measurement Update (Correction)
            kalman_gain = np.where(valid_mask, p_pred / (p_pred + self.r), 0.0)
            
            # Innovation (residual)
            y_t = np.where(valid_mask, z_t - x_pred, 0.0)

            # Update estimate
            x_est = x_pred + kalman_gain * y_t
            p_est = (1.0 - kalman_gain) * p_pred

            denoised[t] = x_est

        if not is_4d and seq.shape[1] == 1:
            denoised = denoised.squeeze(axis=1)

        return denoised

    def filter_single_frame(self, frame: np.ndarray, cloud_mask: np.ndarray = None) -> np.ndarray:
        """
        Denoise single frame (t=1 fallback) using spatial/statistical estimation.
        """
        seq = frame[np.newaxis, ...]
        mask_seq = cloud_mask[np.newaxis, ...] if cloud_mask is not None else None
        res = self.filter_sequence(seq, mask_seq)
        return res[0]
