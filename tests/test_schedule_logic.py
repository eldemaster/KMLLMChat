import unittest

from src.knowledge_manager import KnowledgeManager
from src.models import Activity, Therapy


class TestScheduleLogic(unittest.TestCase):
    def setUp(self):
        self.km = KnowledgeManager()
        self.km.current_patient_id = "test_patient"
        self.km.therapy = Therapy(patient_id="test_patient", activities=[])

        self.km.therapy.activities.extend([
            Activity(
                activity_id="act_01",
                name="Controllo pressione",
                description="Misurazione giornaliera",
                day_of_week=["Lunedì", "Mercoledì"],
                time="08:00",
                dependencies=[],
            ),
            Activity(
                activity_id="act_02",
                name="Take Heart Meds",
                description="Beta-blockers",
                day_of_week=["Monday"],
                time="09:00",
                dependencies=[],
            ),
            Activity(
                activity_id="act_03",
                name="Terapia temporanea",
                description="Valida solo in un intervallo",
                day_of_week=["Lunedì"],
                time="15:00",
                dependencies=[],
                valid_from="2025-01-01",
                valid_until="2025-01-03",
            ),
        ])

    def test_get_activities_by_day_italian_accent(self):
        monday = self.km.get_activities_by_day("Lunedì")
        self.assertEqual(len(monday), 2)
        names = sorted([act.name for act in monday])
        self.assertEqual(names, ["Controllo pressione", "Take Heart Meds"])

    def test_get_activities_by_day_case_insensitive(self):
        monday = self.km.get_activities_by_day("lunedi")
        self.assertEqual(len(monday), 2)
        names = sorted([act.name for act in monday])
        self.assertEqual(names, ["Controllo pressione", "Take Heart Meds"])

    def test_get_activities_by_day_english(self):
        monday = self.km.get_activities_by_day("Monday")
        self.assertEqual(len(monday), 2)
        names = sorted([act.name for act in monday])
        self.assertEqual(names, ["Controllo pressione", "Take Heart Meds"])

    def test_get_activities_by_day_cross_language(self):
        monday = self.km.get_activities_by_day("Lunedì")
        names = sorted([act.name for act in monday])
        self.assertEqual(names, ["Controllo pressione", "Take Heart Meds"])

    def test_get_activities_by_day_empty(self):
        sunday = self.km.get_activities_by_day("Domenica")
        self.assertEqual(sunday, [])

    def test_get_activities_by_day_validity_window(self):
        inside = self.km.get_activities_by_day("Lunedì", date_str="2025-01-02")
        names_inside = sorted([act.name for act in inside])
        self.assertIn("Terapia temporanea", names_inside)

        outside = self.km.get_activities_by_day("Lunedì", date_str="2025-01-05")
        names_outside = sorted([act.name for act in outside])
        self.assertNotIn("Terapia temporanea", names_outside)


if __name__ == "__main__":
    unittest.main()
