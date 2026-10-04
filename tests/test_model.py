import pytest
from src.training.train import build_model, DEFAULT_PARAMS
from src.features.preprocessing import PREPROCESS_FNS

@pytest.mark.parametrize("backbone", PREPROCESS_FNS.keys())
def test_build_model_output_shape(backbone):
    # build_model takes a full params dict, so override only the backbone (4 classes come from the defaults)
    model = build_model({**DEFAULT_PARAMS, "backbone": backbone})

    # Keras output shapes are tuples where the batch dimension is None
    assert model.output_shape == (None, DEFAULT_PARAMS["num_classes"])
