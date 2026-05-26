from tool_relevance_lab.torch_sanity import torch_environment


def test_torch_imports_and_reports_device() -> None:
    env = torch_environment()

    assert env.version
    assert env.device in {"cpu", "cuda"}
