import unittest

from src.main import _apply_duration


class TestDurationLogic(unittest.TestCase):
    def test_apply_duration_with_start_date(self):
        valid_from, valid_until = _apply_duration("2025-01-01", None, 3)
        self.assertEqual(valid_from, "2025-01-01")
        self.assertEqual(valid_until, "2025-01-04")


if __name__ == "__main__":
    unittest.main()
