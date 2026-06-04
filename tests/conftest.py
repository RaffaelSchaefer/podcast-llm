import os

import pytest


@pytest.fixture(autouse=True)
def isolate_podcast_llm_env():
    env_names = (
        "PODCAST_LLM_HOST",
        "PODCAST_LLM_MODEL",
    )
    previous = {name: os.environ.get(name) for name in env_names}
    for name in env_names:
        os.environ.pop(name, None)

    yield

    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
