"""Shared pytest fixtures for FastDepth tests."""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import pytest
import numpy as np
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Provide a mock configuration object."""
    config = Mock()
    config.data_path = '/mock/data/path'
    config.batch_size = 8
    config.input_height = 224
    config.input_width = 224
    config.max_depth = 10.0
    config.min_depth = 0.1
    return config


@pytest.fixture
def sample_rgb_image():
    """Generate a sample RGB image as numpy array."""
    return np.random.rand(224, 224, 3).astype(np.float32)


@pytest.fixture
def sample_depth_map():
    """Generate a sample depth map as numpy array."""
    return np.random.rand(224, 224).astype(np.float32) * 10.0


@pytest.fixture
def sample_batch():
    """Generate a sample batch of RGB images and depth maps."""
    batch_size = 4
    rgb_batch = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    depth_batch = np.random.rand(batch_size, 1, 224, 224).astype(np.float32) * 10.0
    return {
        'rgb': torch.from_numpy(rgb_batch),
        'depth': torch.from_numpy(depth_batch)
    }


@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""
    model = Mock()
    model.eval = Mock(return_value=None)
    model.train = Mock(return_value=None)
    model.parameters = Mock(return_value=[])
    model.state_dict = Mock(return_value={})
    
    def forward_mock(x):
        batch_size = x.shape[0]
        return torch.rand(batch_size, 1, 224, 224)
    
    model.forward = Mock(side_effect=forward_mock)
    model.__call__ = Mock(side_effect=forward_mock)
    return model


@pytest.fixture
def mock_dataloader():
    """Provide a mock dataloader."""
    dataloader = Mock()
    dataloader.__iter__ = Mock(return_value=iter([
        {
            'rgb': torch.rand(4, 3, 224, 224),
            'depth': torch.rand(4, 1, 224, 224)
        }
    ]))
    dataloader.__len__ = Mock(return_value=10)
    return dataloader


@pytest.fixture
def device():
    """Provide appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def mock_checkpoint(temp_dir):
    """Create a mock checkpoint file."""
    checkpoint_path = temp_dir / 'test_checkpoint.pth'
    checkpoint_data = {
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'epoch': 10,
        'loss': 0.5
    }
    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def mock_hdf5_file(temp_dir):
    """Create a mock HDF5 file for testing data loading."""
    import h5py
    
    file_path = temp_dir / 'test_data.h5'
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('rgb', data=np.random.rand(10, 224, 224, 3))
        f.create_dataset('depth', data=np.random.rand(10, 224, 224))
    
    return file_path


@pytest.fixture(autouse=True)
def reset_torch_state():
    """Reset PyTorch state between tests."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured_output)
    return captured_output