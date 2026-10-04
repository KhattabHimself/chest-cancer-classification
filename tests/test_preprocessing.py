import numpy as np
import pytest
from src.features.preprocessing import PREPROCESS_FNS

@pytest.mark.parametrize("backbone_name, preprocess_fn", PREPROCESS_FNS.items())
def test_preprocessing_returns_correct_shape(backbone_name, preprocess_fn):
    # Create a dummy batch of 1 image, 224x224 RGB
    dummy_image = np.zeros((1, 224, 224, 3), dtype=np.float32)
    
    output = preprocess_fn(dummy_image)
    
    # Preprocessing should maintain the spatial and channel dimensions
    assert output.shape == (1, 224, 224, 3)