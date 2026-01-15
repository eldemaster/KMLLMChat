import unittest
from unittest.mock import MagicMock, patch
import json
import os
import tempfile
from pathlib import Path

from src import knowledge_manager as km_mod
from src.knowledge_manager import KnowledgeManager
from src.models import Activity, PatientProfile, CaregiverProfile, Note

class TestPDFScenarios(unittest.TestCase):
    def setUp(self):
        # Setup KnowledgeManager with test data
        self.km = KnowledgeManager()
        # Mock data loading to avoid messing with real files
        self.km.current_patient_id = "test_patient"
        self.km.patient_profile = PatientProfile(
            patient_id="test_patient", 
            name="Test Patient",
            medical_conditions=["Diabete"],
            preferences=["Riposino 15:00"],
            habits=["Caffè dopo pranzo"],
            notes=[Note(content="Aulin = granulare", day=None)]
        )
        # Initialize therapy manually for testing since we are bypassing file loading
        from src.models import Therapy
        self.km.therapy = Therapy(patient_id="test_patient", activities=[])
        
        # Helper to add a base activity
        self.physio = Activity(
            activity_id="act_001", 
            name="Fisioterapia", 
            description="Esercizi gambe", 
            day_of_week=["Lunedì"], 
            time="10:00-11:00", 
            dependencies=[]
        )
        self.km.add_activity(self.physio, force=True)

        self.pill_a = Activity(
            activity_id="act_002", 
            name="Prendere pastiglia A", 
            description="Antipertensivo", 
            day_of_week=["Lunedì"], 
            time="08:00", 
            dependencies=[]
        )
        self.km.add_activity(self.pill_a, force=True)

        self.pill_b = Activity(
            activity_id="act_003", 
            name="Prendere pastiglia B", 
            description="Integratore", 
            day_of_week=["Lunedì"], 
            time="08:30", 
            dependencies=["Prendere pastiglia A"]
        )
        self.km.add_activity(self.pill_b, force=True)

    def test_scenario_1_temporal_conflict(self):
        """
        Scenario: Caregiver adds concurrent activity.
        Existing: Fisioterapia 10:00-11:00
        New: Visita infermieristica 10:30-11:00
        Expected: Conflict detected.
        """
        new_act = Activity(
            activity_id="act_new", 
            name="Visita infermieristica", 
            description="Controllo", 
            day_of_week=["Lunedì"], 
            time="10:30-11:00", 
            dependencies=[]
        )
        conflicts = self.km.check_temporal_conflict(new_act)
        self.assertTrue(len(conflicts) > 0)
        self.assertIn("Fisioterapia", conflicts[0])

    def test_scenario_2_dependency_conflict(self):
        """
        Scenario: Caregiver removes precondition.
        Action: Remove 'Prendere pastiglia A'
        Constraint: 'Prendere pastiglia B' depends on 'Prendere pastiglia A'
        Expected: Warning about dependency.
        """
        conflicts = self.km.check_removal_conflict(self.pill_a)
        self.assertTrue(len(conflicts) > 0)
        self.assertIn("Prendere pastiglia B", conflicts[0])

    def test_scenario_3_semantic_indirect_conflict(self):
        """
        Scenario: "No liquids" vs "Meds with water".
        This requires mocking the LLM check_semantic_conflict function logic 
        since we are testing the integration flow, not the LLM itself here.
        """
        # We simulate that the LLM detects the conflict
        with patch('src.main.check_semantic_conflict') as mock_check:
            mock_check.return_value = "YES: Contradiction detected between 'No liquids' and 'with water'"
            
            # Importing the tool function to test the flow
            from src.main import add_activity_tool
            
            # Setup KM in main (since main.py instantiates its own KM)
            import src.main
            src.main.km = self.km 
            
            # Define a rule in patient profile
            self.km.patient_profile.notes.append(Note(content="Non assumere liquidi per 24 ore", day="Lunedì"))
            
            result = add_activity_tool(
                name="Assunzione farmaci",
                description="Con abbondante acqua",
                days=["Lunedì"],
                time="12:00",
                confirm=True
            )
            
            self.assertIn("BLOCCO SEMANTICO", result)
            self.assertIn("Contradiction", result)

    def test_scenario_4_ambiguity_resolution(self):
        """
        Scenario: "Aulin" defined as "granulare".
        """
        # Verify the note exists
        notes_content = [n.content for n in self.km.patient_profile.notes]
        self.assertIn("Aulin = granulare", notes_content)
        
        # If we were to use the LLM, we would feed this note into the context.
        # Here we just verify the data is present in the profile which is passed to the LLM.
        self.assertEqual(self.km.patient_profile.notes[0].content, "Aulin = granulare")

    def test_scenario_5_activity_update_substitution(self):
        """
        Scenario: update activity due to external conditions.
        """
        activity = Activity(
            activity_id="act_walk",
            name="Camminata",
            description="Uscire a fare una camminata",
            day_of_week=["Lunedì"],
            time="18:00",
            dependencies=[],
        )
        self.km.add_activity(activity, force=True)
        result = self.km.update_activity(
            "Camminata",
            "Lunedì",
            {"name": "Cyclette", "description": "Fare cyclette al chiuso"},
            force=True,
        )
        self.assertIn("successo", result.lower())
        updated = self.km.get_activity_by_name_day("Cyclette", "Lunedì")
        self.assertIsNotNone(updated)

    def test_scenario_6_caregiver_info_and_ambiguity(self):
        """
        Scenario: caregiver knowledge updates and semantic ambiguity.
        """
        tmp = tempfile.TemporaryDirectory()
        original_data_dir = km_mod.DATA_DIR
        try:
            data_dir = Path(tmp.name)
            for sub in ("patients", "caregivers", "therapies"):
                (data_dir / sub).mkdir(parents=True, exist_ok=True)
            km_mod.DATA_DIR = data_dir

            self.km.current_caregiver_id = "caregiver_test"
            self.km.caregiver_profile = CaregiverProfile(
                caregiver_id="caregiver_test",
                name="Andrea",
                notes=[],
                semantic_preferences=[],
            )

            res1 = self.km.save_knowledge_note("caregiver", "Visita abituale alle 18:00")
            res2 = self.km.save_knowledge_note("caregiver", "Oggi visita anticipata alle 14:00")
            res3 = self.km.save_knowledge_note("caregiver", "Aulin = granulare")
            res4 = self.km.save_knowledge_note("caregiver", "Aulin = supposta")

            self.assertIn("Nota salvata", res1)
            self.assertIn("Nota salvata", res2)
            self.assertIn("Nota salvata", res3)
            self.assertIn("Nota salvata", res4)
            self.assertEqual(len(self.km.caregiver_profile.notes), 4)
        finally:
            km_mod.DATA_DIR = original_data_dir
            tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
