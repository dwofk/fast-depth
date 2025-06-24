"""Validation tests to verify the testing infrastructure is properly set up."""

import sys
from pathlib import Path
import pytest


class TestInfrastructureSetup:
    """Test class to validate the testing infrastructure."""
    
    @pytest.mark.unit
    def test_pytest_is_available(self):
        """Test that pytest is properly installed."""
        assert 'pytest' in sys.modules or pytest is not None
        
    @pytest.mark.unit
    def test_testing_directory_structure(self):
        """Test that the testing directory structure is correct."""
        test_root = Path(__file__).parent
        
        assert test_root.exists()
        assert test_root.name == 'tests'
        
        assert (test_root / '__init__.py').exists()
        assert (test_root / 'conftest.py').exists()
        assert (test_root / 'unit').exists()
        assert (test_root / 'unit' / '__init__.py').exists()
        assert (test_root / 'integration').exists()
        assert (test_root / 'integration' / '__init__.py').exists()
        
    @pytest.mark.unit
    def test_fixtures_are_available(self, temp_dir, mock_config):
        """Test that fixtures from conftest.py are available."""
        assert temp_dir.exists()
        assert hasattr(mock_config, 'batch_size')
        assert mock_config.batch_size == 8
        
    @pytest.mark.unit
    def test_coverage_is_configured(self):
        """Test that coverage is properly configured."""
        try:
            import coverage
            assert coverage is not None
        except ImportError:
            pytest.fail("Coverage module is not installed")
            
    @pytest.mark.unit
    def test_mock_is_available(self):
        """Test that pytest-mock is available."""
        try:
            from pytest_mock import MockerFixture
            assert MockerFixture is not None
        except ImportError:
            pytest.fail("pytest-mock is not installed")
            
    @pytest.mark.unit
    def test_markers_are_defined(self, request):
        """Test that custom markers are properly defined."""
        markers = request.config.getini('markers')
        assert any('unit' in marker for marker in markers)
        assert any('integration' in marker for marker in markers)
        assert any('slow' in marker for marker in markers)
        
    @pytest.mark.unit
    def test_torch_fixtures(self, sample_batch, device):
        """Test that PyTorch-related fixtures work correctly."""
        assert 'rgb' in sample_batch
        assert 'depth' in sample_batch
        assert sample_batch['rgb'].shape == (4, 3, 224, 224)
        assert sample_batch['depth'].shape == (4, 1, 224, 224)
        assert device is not None
        
    @pytest.mark.integration
    def test_temp_file_operations(self, temp_dir):
        """Test that temporary file operations work in fixtures."""
        test_file = temp_dir / 'test.txt'
        test_file.write_text('test content')
        
        assert test_file.exists()
        assert test_file.read_text() == 'test content'
        
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker is properly configured."""
        import time
        time.sleep(0.1)
        assert True


def test_poetry_scripts_configured():
    """Test that poetry scripts are properly configured."""
    pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'
    assert pyproject_path.exists()
    
    content = pyproject_path.read_text()
    assert '[tool.poetry.scripts]' in content
    assert 'test = "pytest:main"' in content
    assert 'tests = "pytest:main"' in content