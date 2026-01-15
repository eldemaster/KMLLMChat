import unittest
from src.knowledge_manager import KnowledgeManager
from src.models import Activity, PatientProfile, CaregiverProfile, Note, Therapy

class TestAdvancedLogic(unittest.TestCase):
    def setUp(self):
        self.km = KnowledgeManager()
        # Mocking Context
        self.km.patient_profile = PatientProfile(patient_id="test_p", name="Mario", medical_conditions=[])
        self.km.caregiver_profile = CaregiverProfile(caregiver_id="test_c", name="Maria")
        self.km.therapy = Therapy(patient_id="test_p", activities=[])

        # Base Activity: Surgery
        self.surgery = Activity(
            activity_id="act_surg",
            name="Chirurgia",
            description="Intervento",
            day_of_week=["Lunedì"],
            time="08:00",
            dependencies=[]
        )
        self.km.therapy.activities.append(self.surgery)

    def test_missing_dependency_on_add(self):
        """
        Scenario: Provo ad aggiungere 'Riabilitazione' che dipende da 'Ortopedico',
        ma 'Ortopedico' non è in terapia.
        """
        rehab = Activity(
            activity_id="act_rehab",
            name="Riabilitazione",
            description="Post intervento",
            day_of_week=["Lunedì"],
            time="14:00",
            dependencies=["Ortopedico"] # Dipendenza inesistente
        )
        
        # Check logic directly
        issues = self.km.check_missing_dependencies(rehab)
        self.assertTrue(len(issues) > 0)
        self.assertIn("non trovata", issues[0])

    def test_dependency_sequence_error(self):
        """
        Scenario: 'Riabilitazione' dipende da 'Chirurgia', ma la schedulo PRIMA della chirurgia.
        Chirurgia: 08:00
        Riabilitazione: 07:00 (Errore logico)
        """
        rehab_early = Activity(
            activity_id="act_rehab_bad",
            name="Riabilitazione Presto",
            description="...",
            day_of_week=["Lunedì"],
            time="07:00",
            dependencies=["Chirurgia"]
        )
        
        issues = self.km.check_missing_dependencies(rehab_early)
        self.assertTrue(len(issues) > 0)
        self.assertIn("inizia prima", issues[0])

    def test_modification_substitution(self):
        """
        Scenario PDF: Sostituzione attività.
        Modifico 'Chirurgia' in 'Terapia Conservativa'.
        """
        # Simuliamo la modifica chiamando update_activity
        # (Nota: update_activity usa gli ID reali o i nomi per trovare l'attività)
        
        result = self.km.update_activity(
            old_name="Chirurgia",
            day="Lunedì",
            new_data={"name": "Terapia Conservativa", "description": "Niente operazione"},
            force=False
        )
        
        self.assertIn("successo", result)
        # Verify change
        act = self.km.therapy.activities[0]
        self.assertEqual(act.name, "Terapia Conservativa")
        self.assertEqual(act.description, "Niente operazione")

if __name__ == '__main__':
    unittest.main()
