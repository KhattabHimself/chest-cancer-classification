import pytest
from src.models.model import build_model
from src.features.preprocessing import PREPROCESS_FNS

@pytest.mark.parametrize("backbone", PREPROCESS_FNS.keys())
def test_build_model_output_shape(backbone):
    # Pass the backbone name and force 4 classes for the chest cancer categories
    model = build_model(backbone=backbone, num_classes=4)
    
    # Keras output shapes are tuples where the batch dimension is None
    assert model.output_shape == (None, 4)