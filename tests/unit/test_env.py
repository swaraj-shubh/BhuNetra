def test_ml_stack_imports_with_matching_builds():
    """torchvision refuses to import if its CUDA build differs from torch's (pyproject tool.uv.sources)."""
    import open_clip  # noqa: F401  (imports torchvision)
    import torch
    import torchvision

    assert torch.__version__.split("+")[1:] == torchvision.__version__.split("+")[1:], (
        torch.__version__,
        torchvision.__version__,
    )
