import pytest
from src.knowledge_manager import KnowledgeManager
from src.models import Activity, Therapy

@pytest.fixture
def km_with_deps():
    """KnowledgeManager con attività collegate da dipendenze"""
    km = KnowledgeManager()
    
    act1 = Activity(
        activity_id="1", name="Colazione", description="Mangiare", 
        day_of_week=["Lunedì"], time="08:00", dependencies=[]
    )
    act2 = Activity(
        activity_id="2", name="Farmaco A", description="Prendere dopo colazione", 
        day_of_week=["Lunedì"], time="08:30", dependencies=["Colazione"]
    )
    
    km.therapy = Therapy(patient_id="test", activities=[act1, act2])
    return km

def test_remove_dependency_conflict(km_with_deps):
    # Provo a rimuovere "Colazione". "Farmaco A" dipende da essa.
    # Mi aspetto che il sistema rilevi che "Farmaco A" rimarrebbe orfano.
    
    # Cerco l'attività da rimuovere
    act_to_remove = next(a for a in km_with_deps.therapy.activities if a.name == "Colazione")
    
    conflicts = km_with_deps.check_removal_conflict(act_to_remove)
    assert len(conflicts) > 0
    assert "Farmaco A" in conflicts[0]

def test_remove_safe(km_with_deps):
    # Rimuovo "Farmaco A". Nessuno dipende da lui.
    act_to_remove = next(a for a in km_with_deps.therapy.activities if a.name == "Farmaco A")
    
    conflicts = km_with_deps.check_removal_conflict(act_to_remove)
    assert len(conflicts) == 0

def test_add_missing_dependency(km_with_deps):
    # Aggiungo attività che dipende da "Pranzo" (che non c'è)
    new_act = Activity(
        activity_id="3", name="Caffè", description="Dopo pranzo", 
        day_of_week=["Lunedì"], time="13:00", dependencies=["Pranzo"]
    )
    
    # Nota: check_temporal_conflict non controlla dipendenze. 
    # Dovremo aggiungere un metodo check_dependency_completeness o integrarlo in add_activity
    issues = km_with_deps.check_missing_dependencies(new_act)
    assert len(issues) > 0
    assert "Pranzo" in issues[0]
