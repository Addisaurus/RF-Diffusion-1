import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def test_basic():
    """Basic test to verify pytest is working"""
    assert True

def test_torch_import():
    """Test that we can import torch"""
    try:
        import torch
        assert True
    except ImportError:
        assert False, "Failed to import torch"

def test_numpy_import():
    """Test that we can import numpy"""
    try:
        import numpy as np
        assert True
    except ImportError:
        assert False, "Failed to import numpy"

def test_torch_operations():
    """Test basic torch operations"""
    import torch
    x = torch.randn(2, 2)
    y = x + x
    assert y.shape == (2, 2)
    
def test_device_available():
    """Test CUDA availability"""
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    assert True