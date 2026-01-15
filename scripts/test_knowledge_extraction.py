import unittest


@unittest.skip("LLM-based extraction is tested manually.")
class TestKnowledgeExtraction(unittest.TestCase):
    def test_placeholder(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
