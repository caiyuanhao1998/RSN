"""Ensure CUDA variants are audited and floating dependency versions fail."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_requirements.py'
spec = importlib.util.spec_from_file_location('audit_requirements', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_audit_includes_framework_and_cuda_dependencies():
    content = ('torch==2.13.0+cu126\ntorchvision==0.28.0+cu126\n'
               'nvidia-cublas-cu12==12.6.4.1\nnumpy==1.26.4\n')
    assert module.normalize_requirements(content) == (
        'torch==2.13.0\ntorchvision==0.28.0\n'
        'nvidia-cublas-cu12==12.6.4.1\nnumpy==1.26.4\n')


@pytest.mark.parametrize('content', ['torch>=2.13', 'torch==2.*', '-r other.txt', '# empty\n'])
def test_audit_rejects_incomplete_dependency_collection(content):
    with pytest.raises(ValueError):
        module.normalize_requirements(content)
