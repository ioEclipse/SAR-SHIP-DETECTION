"""
SAR Ship Tracking Module

This module contains functions and classes for tracking ships across
sequential SAR images using various tracking algorithms.

Available modules:
- noise_filter: Noise filtering for tracking
- Land_masking: Land masking for tracking applications
- image_preprocessing: Image preprocessing for tracking pipeline
"""

# Import key functions when available
try:
    from .noise_filter import apply_correction
    from .Land_masking import process_image, compare_images
    __all__ = ['apply_correction', 'process_image', 'compare_images']
except ImportError:
    # If imports fail, just define empty __all__
    __all__ = []