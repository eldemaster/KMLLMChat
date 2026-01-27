import unittest

from src.main import _extract_json_object, _parse_action_string, _sanitize_tool_args


class TestToolParsing(unittest.TestCase):
    def test_extract_json_object(self):
        text = "foo {\"action\": \"get_schedule\", \"day\": \"Lunedì\"} bar"
        data = _extract_json_object(text)
        self.assertEqual(data["action"], "get_schedule")
        self.assertEqual(data["day"], "Lunedì")

    def test_parse_action_string(self):
        text = "modify_activity(old_name=Passeggiata, day=Lunedì, new_name=Passeggiata, new_description=null, new_time=13:00, force=false)"
        data = _parse_action_string(text)
        self.assertEqual(data["tool_name"], "modify_activity")
        args = data["arguments"]
        self.assertEqual(args["old_name"], "Passeggiata")
        self.assertEqual(args["day"], "Lunedì")
        self.assertIsNone(args["new_description"])
        self.assertEqual(args["new_time"], "13:00")
        self.assertFalse(args["force"])

    def test_parse_action_string_with_list(self):
        text = "add_activity(name=Controllo, description=Test, days=[\"Lunedì\",\"Mercoledì\"], time=09:00, dependencies=[])"
        data = _parse_action_string(text)
        args = data["arguments"]
        self.assertEqual(args["days"], ["Lunedì", "Mercoledì"])
        self.assertEqual(args["dependencies"], [])

    def test_sanitize_add_activity_interval_to_duration(self):
        args = {"name": "Test", "days": ["Lunedì"], "time": "10:00-10:30"}
        cleaned = _sanitize_tool_args("add_activity", args)
        self.assertEqual(cleaned["time"], "10:00")
        self.assertEqual(cleaned["duration_minutes"], 30)

    def test_sanitize_modify_activity_new_days_and_interval(self):
        args = {
            "old_name": "Passeggiata",
            "day": "Lunedì",
            "new_days": ["Lunedì", "Giovedì"],
            "new_time": "10:00-10:45",
        }
        cleaned = _sanitize_tool_args("modify_activity", args)
        self.assertEqual(cleaned["new_days"], ["Lunedì", "Giovedì"])
        self.assertEqual(cleaned["new_time"], "10:00")
        self.assertEqual(cleaned["duration_minutes"], 45)

if __name__ == "__main__":
    unittest.main()
