"""No compiled extension or torch required: python -m unittest discover ..."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "mxfp4_reuse", Path(__file__).resolve().parents[1] / "python/utils/mxfp4_reuse.py"
)
options = importlib.util.module_from_spec(spec)
spec.loader.exec_module(options)


class TokenReuseConfigTest(unittest.TestCase):
    def test_defaults_and_other_formats(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(options.token_reuse_enabled())
            self.assertEqual(options.group_min_len((39, 39, 39)), 2)
            self.assertEqual(options.group_min_len((39, 39, 39), native=True), 10)
            self.assertEqual(options.group_min_len((39, 8, 39)), 10)

    def test_rollback_and_threshold_override(self):
        with patch.dict(os.environ, {"KT_MOE_MXFP4_TOKEN_REUSE": "0"}, clear=True):
            self.assertFalse(options.token_reuse_enabled())
            self.assertEqual(options.group_min_len((39, 39, 39)), 10)
            os.environ["KT_GROUP_MIN_LEN"] = "2"
            self.assertEqual(options.group_min_len((39, 39, 39)), 2)

    def test_invalid_options(self):
        for value in ("", "true", "2"):
            with patch.dict(os.environ, {"KT_MOE_MXFP4_TOKEN_REUSE": value}, clear=True):
                with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                    options.token_reuse_enabled()
        for value in ("", "bad", "0", "-1"):
            with patch.dict(os.environ, {"KT_GROUP_MIN_LEN": value}, clear=True):
                with self.assertRaises(ValueError):
                    options.group_min_len((39, 39, 39))


if __name__ == "__main__":
    unittest.main()
