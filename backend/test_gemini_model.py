import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("main.py")


spec = importlib.util.spec_from_file_location("quickquiz_main", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_default_gemini_model_is_supported():
    assert module.GEMINI_MODEL == "gemini-2.5-flash"
