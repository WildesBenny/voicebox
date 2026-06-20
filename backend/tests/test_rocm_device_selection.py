import types

from backend.backends.base import get_torch_device


class FakeTorch(types.SimpleNamespace):
    pass


def test_get_torch_device_prefers_cuda_for_rocm(monkeypatch):
    fake_torch = FakeTorch(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        version=types.SimpleNamespace(hip="7.2"),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )

    monkeypatch.setitem(__import__('sys').modules, 'torch', fake_torch)

    assert get_torch_device() == 'cuda'
