import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def test_imports():
    """Test that we can import required modules without errors"""
    import torch
    import numpy as np
    from tfdiff.conditioning import ConditioningManager
    assert True

def test_torch_available():
    """Test that PyTorch is working"""
    import torch
    x = torch.randn(2, 2)
    assert x.shape == (2, 2)
    print("PyTorch test passed!")